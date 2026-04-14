# Factory Ledger — Full Project Context

> Paste this document into a new Claude conversation to give it complete context about the Factory Ledger system. This is a comprehensive reference — not a summary.

---

## What Is This?

Factory Ledger is a **manufacturing inventory management and production scheduling system** for a food manufacturing plant (CNS — granola, coconut products, retail snacks). It tracks raw materials in → production batches → finished goods out, with full lot-level traceability, sales order fulfillment, and 7-day tactical production scheduling.

**Stack:** Python 3.11.7 / FastAPI 0.104.1 / PostgreSQL (Supabase) / Vanilla JS dashboard
**Deployment:** Railway (API) + Netlify (dashboard static files)
**Version:** 2.5.0
**Dependencies:** fastapi, uvicorn, psycopg2-binary, python-multipart, aiofiles

---

## File Structure

```
factory-ledger/
├── main.py                       # ALL backend logic (6,624 lines, single-file FastAPI app)
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python 3.11.7
├── dashboard/
│   ├── index.html                # Dashboard HTML (tabs: Operations, Activity, Notes, Sales Orders)
│   ├── dashboard.js              # Vanilla JS dashboard app (1,293 lines, no framework)
│   ├── dashboard.css             # Dark/light theme CSS (1,137 lines)
│   └── dashboard_config.json     # SKU panel definitions, batch sizes, ingredient categories
├── migrations/
│   ├── 003_notes_todos_reminders.sql
│   └── 004_production_scheduling.sql
├── openapi-schema.yaml           # Full OpenAPI 3.0 spec
├── openapi-schema-gpt.yaml       # GPT-optimized OpenAPI spec (for ChatGPT custom GPT integration)
├── GUIDE.md                      # User workflow guide (21 operations with real-world examples)
├── SALES_API.md                  # Sales & customer API reference with curl examples
└── DEPLOYMENT.md                 # Deployment & setup guide
```

**Key architectural fact:** The entire backend is a single `main.py` file. All routes, Pydantic models, database migrations, helper functions, and business logic live in one file. There is no ORM — all database access is raw SQL via psycopg2.

---

## Database Architecture

### Connection Management

```python
db_pool = pool.ThreadedConnectionPool(minconn=2, maxconn=20, dsn=DATABASE_URL)

@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)

@contextmanager
def get_transaction():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
```

All queries use `RealDictCursor` for dict-style row access. Write operations use `get_db_connection()` directly (for `FOR UPDATE` locking), read operations use `get_transaction()`.

### Core Tables

**`products`** — Product registry
- `id` SERIAL PK, `name` TEXT, `type` TEXT ('ingredient'|'batch'|'finished'), `uom` TEXT ('lb'|'unit')
- `odoo_code` TEXT (ERP integration code), `active` BOOLEAN
- `default_batch_lb` NUMERIC (standard batch weight for batch products)
- `default_case_weight_lb` NUMERIC (weight per case for finished goods)
- `case_size_lb` NUMERIC(10,2) (sellable unit weight, e.g., 25 for "25 LB" case)
- `yield_multiplier` FLOAT DEFAULT 1.0 (weight change during processing)
- `label_type` TEXT DEFAULT 'house' ('house'|'private_label') — SKU protection flag
- `storage_type` TEXT ('ambient'|'refrigerated')
- `verification_status` TEXT ('unverified'|'verified')
- `parent_product_id` INTEGER (for rapid merging)
- `notes`, `notes_es` TEXT (bilingual)
- Indexes on: name, odoo_code, type, active

**`lots`** — Lot tracking
- `id` SERIAL PK, `product_id` INTEGER FK→products
- `lot_code` TEXT UNIQUE (format: `YY-MM-DD-XXXX-###` for received, `BYYMM-DD-###` for batches)
- `shipper_name`, `shipper_code`, `bol_reference` TEXT
- `entry_source` TEXT ('received'|'production_output'|'pack_output'|'found')
- `created_at` TIMESTAMPTZ
- Indexes on: product_id, lot_code

**`transactions`** — Immutable audit log (append-only)
- `id` SERIAL PK, `type` TEXT ('receive'|'ship'|'make'|'pack'|'adjust')
- `timestamp` TIMESTAMPTZ, `performed_by` TEXT
- `customer_name`, `order_reference` TEXT (for ship transactions)
- `shipper_name`, `shipper_code`, `bol_reference` TEXT, `cases_received`, `case_size_lb` (for receive)
- `adjust_reason`, `adjust_reason_es` TEXT (for adjustments)
- `notes`, `notes_es` TEXT (bilingual)

**`transaction_lines`** — Double-entry ledger lines
- `id` SERIAL PK, `transaction_id` FK→transactions, `lot_id` FK→lots, `product_id` FK→products
- `quantity_lb` NUMERIC (positive=inflow, negative=outflow)
- `sequence_number` INTEGER
- Indexes on: transaction_id, lot_id, product_id

**`ingredient_lot_consumption`** — Traceability linkage (which ingredient lots went into which batch)
- `transaction_id` FK→transactions, `ingredient_product_id` FK→products, `ingredient_lot_id` FK→lots
- `quantity_lb` NUMERIC

**`batch_formulas`** — BOM recipes
- `id` SERIAL PK, `product_id` FK→products (batch), `ingredient_product_id` FK→products (ingredient)
- `quantity_lb` NUMERIC (per single batch)
- `exclude_from_inventory` BOOLEAN DEFAULT false (for Water and similar utility ingredients)

**`product_bom`** — Finished good → batch product mapping
- `id` SERIAL PK, `finished_product_id` FK→products, `component_product_id` FK→products
- `quantity` NUMERIC DEFAULT 1.0, `uom` TEXT DEFAULT 'unit'

**`customers`** — Customer registry
- `id` SERIAL PK, `name` TEXT, `contact_name`, `email`, `phone`, `address` TEXT
- `active` BOOLEAN, `notes`, `notes_es` TEXT

**`sales_orders`** — Order headers
- `id` SERIAL PK, `order_number` TEXT UNIQUE (auto-generated)
- `customer_id` FK→customers, `status` TEXT, `requested_ship_date` DATE
- `order_date` DATE, `notes`, `notes_es` TEXT, `created_at` TIMESTAMPTZ
- **Status state machine:** `confirmed → in_production → ready → shipped/partial_ship → invoiced` (also `cancelled`)
- `shipped` and `partial_ship` are set **automatically** by ship/commit — cannot be set manually

**`sales_order_lines`** — Order line items
- `id` SERIAL PK, `sales_order_id` FK→sales_orders, `product_id` FK→products
- `quantity_lb` NUMERIC, `quantity_shipped_lb` NUMERIC DEFAULT 0
- `unit_price` NUMERIC, `line_status` TEXT ('pending'|'partial'|'fulfilled'|'cancelled')
- `notes`, `notes_es` TEXT

**`sales_order_shipments`** — Links shipment transactions to order lines
- `sales_order_line_id` FK→sales_order_lines, `transaction_id` FK→transactions
- `quantity_lb` NUMERIC, `shipped_at` TIMESTAMPTZ

**`notes`** — Notes/todos/reminders
- `id` SERIAL PK, `category` TEXT ('note'|'todo'|'reminder')
- `title` TEXT, `body` TEXT, `priority` TEXT ('low'|'normal'|'high')
- `status` TEXT ('open'|'done'|'dismissed'), `due_date` DATE
- `entity_type` TEXT ('product'|'lot'|'customer'|'supplier'), `entity_id` TEXT
- `created_at`, `updated_at` TIMESTAMPTZ

### Production Scheduling Tables (Migration 004)

**`production_lines`** — 4 lines: Granola Baking, Coconut Sweetened, Bulk Packing, Pouch Line

**`line_capacity_modes`** — Worker configs per line:
- Granola: 2-worker = 9 batches/day (default), 3-worker = 16 batches/day
- Coconut: standard = 12 batches/day (2 workers)
- Bulk Pack: 25lb-cases = 4 pallets/day (2 workers, default), 10lb-cases = 9 pallets/day
- Pouch: standard = 7,500 bags/day (3 workers)

**`product_line_assignments`** — Which products run on which lines

**`production_schedule`** — Confirmed schedule runs (date, line, product, batches, workers, status, linked orders)

**`scheduling_config`** — Global params: total_workers=10, friday_capacity_modifier=0.5, work_days=[Mon-Fri], default_horizon_days=7

### Inventory Model

Inventory is **ledger-based** (not snapshot-based). Current stock = SUM of all `transaction_lines.quantity_lb` for a given lot. Positive quantities are inflows (receive, make, pack-output), negative are outflows (ship, pack-source, make-ingredient-consumption). This is an **append-only** ledger — no updates or deletes to transaction_lines.

**FIFO allocation:** When shipping or packing, the system allocates from the oldest lots first (ORDER BY lot.id ASC). Multi-lot shipments split across lots in FIFO order. Row-level locking (`SELECT ... FOR UPDATE`) prevents race conditions.

### Startup Migrations (7 total, run automatically)

1. **label_type column** — Adds `label_type` to products, flags private-label SKUs by odoo_code and name patterns ('Batch BS %', 'Batch Setton %')
2. **exclude_from_inventory** — Adds to batch_formulas, auto-flags Water entries
3. **yield_multiplier** — Adds to products (DEFAULT 1.0)
4. **case_size_lb** — Adds to products, auto-populates from product names (25 LB → 25, 10 LB → 10, 50 LB → 50)
5. **Legacy order status** — Migrates 'new' orders to 'confirmed'
6. (Plus migration files 003 and 004 for notes and scheduling tables)

---

## API Endpoints (~77 routes)

All routes require `X-API-Key` header except `/dashboard/api/*` endpoints (read-only, no auth).

### Authentication

```python
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return True
```

### Core Transaction Endpoints (Preview/Commit Pattern)

Every inventory-affecting operation uses a two-step pattern:
1. `POST /{operation}/preview` — dry-run showing what will happen (no DB writes)
2. `POST /{operation}/commit` — executes the transaction, creates lots/lines, returns confirmation code `TXN-XXXXXX`

#### RECEIVE — Incoming Goods

```python
class ReceiveRequest(BaseModel):
    product_name: str
    cases: int
    case_size_lb: float
    shipper_name: str
    bol_reference: str
    shipper_code_override: Optional[str] = None
```

- Preview resolves product via `resolve_product_full()`, generates lot code
- Commit creates lot (entry_source='received'), transaction, transaction_line (+quantity)
- Lot code format: `YY-MM-DD-XXXX-###` where XXXX = 4-char shipper code (auto-extracted from name or overridden)
- Total weight = cases * case_size_lb

#### SHIP — Outgoing Shipments

```python
class ShipRequest(BaseModel):
    product_name: str
    quantity_lb: float
    customer_name: str
    order_reference: str
    lot_code: Optional[str] = None  # Pin to specific lot; otherwise FIFO
```

- **Single-lot path:** If specified lot (or FIFO-oldest) has enough inventory, ships from that one lot
- **Multi-lot FIFO fallback:** If single lot insufficient, automatically splits across multiple lots in FIFO order
- Row-level locking: `SELECT ... FOR UPDATE` on lots before checking balances
- **Open orders warning:** If the customer has open sales orders, the response includes a warning suggesting to use the order-based ship endpoint instead
- Commit creates transaction with negative transaction_lines for each lot consumed

#### MAKE — Production Runs

```python
class MakeRequest(BaseModel):
    product_name: str
    batches: int
    lot_code: Optional[str] = None
    ingredient_lot_overrides: Optional[Union[Dict[str, str], str]] = None
    excluded_ingredients: Optional[List[int]] = None
    confirmed_sku: Optional[bool] = None
```

- **Sibling SKU check:** If product shares a BOM with other products (e.g., house brand vs. private label), requires `confirmed_sku: true`
- **Ingredient lot overrides:** Dict mapping ingredient_product_id → lot_code to use specific lots instead of FIFO
- **Excluded ingredients:** List of ingredient IDs to skip (for partial batches or substitutions)
- **Auto-excluded ingredients:** Ingredients flagged `exclude_from_inventory=true` in batch_formulas (e.g., Water) are automatically skipped
- **Output:** Creates batch lot (code format: `BYYMM-DD-###`), positive line for output, negative lines for each consumed ingredient lot
- **Yield multiplier:** Output weight = formula_weight * yield_multiplier
- Records ingredient lot consumption in `ingredient_lot_consumption` table for traceability
- Returns `daily_production_summary` in response

#### PACK — Batch → Finished Good

```python
class PackRequest(BaseModel):
    source_product: str          # Batch product name/code
    target_product: str          # Finished good name/code
    cases: int
    case_weight_lb: Optional[float] = None  # Override; defaults to product's default
    lot_allocations: Optional[List[PackLotAllocation]] = None  # Explicit lot splits
    target_lot_code: Optional[str] = None   # Override output lot code
```

- Resolves both source (batch) and target (finished good) products
- Case weight defaults to target product's `default_case_weight_lb`
- **FIFO allocation** across batch lots if no explicit `lot_allocations` provided
- **Explicit allocations:** If provided, must sum to exactly cases * case_weight_lb
- Output lot code inherits from primary batch lot by default
- Creates positive line for finished good, negative lines for batch lot consumption
- Returns `daily_production_summary`

#### ADJUST — Inventory Corrections

```python
class AdjustRequest(BaseModel):
    product_name: str
    lot_code: str
    adjustment_lb: float  # Positive or negative
    reason: str
    reason_es: Optional[str] = None
```

- **SKU protection:** Blocks destructive adjustments on private-label SKUs if reason contains merge/deprecate/consolidate/migrate keywords AND quantity is negative
- Bilingual validation on reason field
- Preview shows current balance and projected new balance (with warning if going negative)
- Returns 403 with `blocked: True` if SKU protection triggers

### Product Management

- `GET /products/search?q=...` — Fuzzy search by name/odoo_code, returns matches with inventory totals
- `GET /products/unverified` — Products in review queue (verification_status='unverified')
- `GET /products/test-batches` — Test/sample batch products
- `GET /products/{product_id}` — Full product details
- `POST /products/quick-create` — Create product on-the-fly during operations (type, uom, storage_type, name_confidence)
- `POST /products/quick-create-batch` — Create batch product with category and production_context
- `POST /products/{id}/verify` — Verify, reject, or archive products (action: 'verify'|'merge'|etc.)
- `PUT /admin/products/{id}` — Update product fields (default_case_weight_lb, default_batch_lb, yield_multiplier, active)

### Inventory

- `GET /inventory/current` — All products with positive on-hand, grouped by product with lot-level breakdown
- `GET /inventory/{item_name}` — Specific product inventory with lot details
- `POST /inventory/found` — Log found/unmarked inventory (existing product)
- `POST /inventory/found-with-new-product` — Log found inventory + create new product simultaneously
- `GET /inventory/found/queue` — Found inventory awaiting review

### Lots

- `GET /lots/by-code/{lot_code}` — Lot details by code
- `GET /lots/{lot_id}` — Lot details by ID
- `POST /lots/{lot_id}/reassign` — Move lot to different product (with reason code, SKU protection)

### Sales Orders

**Status state machine:**
```
confirmed → in_production → ready → shipped/partial_ship → invoiced
                                  ↘ cancelled (from any non-terminal state)
```
- `shipped` and `partial_ship` are set **automatically** by ship/commit — blocked from manual setting
- Header edits only allowed when status is 'new' or 'confirmed'

**Order creation supports multiple quantity units:**
```python
class OrderLineInput(BaseModel):
    product_name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None  # 'lb', 'cases', 'bags', 'boxes'
    case_weight_lb: Optional[float] = None
    quantity_lb: Optional[float] = None
    unit_price: Optional[float] = None
```
- If unit is 'cases'/'bags'/'boxes' and no case_weight_lb provided, auto-looks up from product's `default_case_weight_lb`
- Warns if quantity provided without explicit unit (defaults to lb)
- Sanity-checks quantity against customer's average order size

**Endpoints:**
- `POST /sales/orders` — Create order (auto-generates order_number, status starts as 'confirmed')
- `GET /sales/orders` — List with filters (status, customer, overdue_only, limit)
- `GET /sales/orders/{id}` — Full detail with lines, shipments history, value calculations
- `GET /sales/orders/fulfillment-check` — Which orders can be fulfilled with current inventory (fulfillable/partially_fulfillable/blocked)
- `PATCH /sales/orders/{id}/status` — Transition status (enforced state machine)
- `PATCH /sales/orders/{id}` — Update header (ship date, notes, customer) — only when confirmed
- `POST /sales/orders/{id}/lines` — Add lines to existing order
- `PATCH /sales/orders/{id}/lines/{line_id}/cancel` — Cancel specific line
- `PATCH /sales/orders/{id}/lines/{line_id}/update` — Update quantity_lb or unit_price
- `POST /sales/orders/{id}/ship/preview` — Preview shipping (shows on-hand vs. needed per line)
- `POST /sales/orders/{id}/ship/commit` — Execute shipment with FIFO lot allocation per line, auto-updates line_status and order status

**Price calculation:** `line_value = cases * unit_price` (not lb * price). For products without case_size_lb, falls back to `lb * price`. Non-weight items (pallets, freight, surcharges) detected by keyword and priced per_unit.

### Traceability

- `GET /trace/batch/{lot_code}` — **Backward trace:** What ingredient lots went into this batch? Uses `ingredient_lot_consumption` table.
- `GET /trace/ingredient/{lot_code}` — **Forward trace:** What finished products used this ingredient lot?

### Production Scheduling

`POST /schedule` — Unified endpoint with `action` dispatch:

**`suggest` action:**
1. Loads open/confirmed sales orders within horizon (default 7 working days)
2. Loads current finished goods and batch inventory
3. Runs simulated allocation: demand in ship-date order, allocate from inventory, calculate net production needs
4. Maps finished goods → batch products via `product_bom`, calculates batches needed (rounded up)
5. Checks batch inventory too — deducts any on-hand batch product before calculating production
6. Assigns runs to days respecting line capacity modes, worker constraints (default 10), and Friday 50% modifier
7. Explodes ingredients: calculates total ingredient needs across all scheduled batches, flags shortages
8. Returns scenarios with day-by-day line assignments, overproduction tracking (with "Batch Rounding" reason), and unscheduled orders

**`confirm` action:** Saves schedule runs to `production_schedule` table

**`current` action:** Returns the current confirmed schedule

### BOM Management

- `GET /bom/products` — List products that have BOM formulas (searchable)
- `GET /bom/batches/{batch_id}/formula` — Get recipe for a batch product (ingredients + quantities)
- `GET /admin/bom/search` — Search products for BOM admin UI
- `GET /admin/bom/{product_id}/lines` — List BOM ingredient lines
- `POST /admin/bom/{product_id}/lines` — Add BOM ingredient line
- `PUT /admin/bom/lines/{line_id}` — Update BOM line (quantity, exclude_from_inventory)
- `DELETE /admin/bom/lines/{line_id}` — Delete BOM line
- `GET /admin/product-bom` — List FG→batch mappings
- `POST /admin/product-bom` — Create FG→batch mapping
- `DELETE /admin/product-bom/{mapping_id}` — Delete mapping

### Production Analysis

- `GET /production/requirements?product_name=...&cases=N` — Full ingredient breakdown for a finished good (supports nested BOMs where an ingredient is itself a batch product)
- `GET /production/day-summary?date=YYYY-MM-DD` — All make/pack/adjust activity for a day, grouped by product with lot-level detail

### Dashboard API (No Auth)

- `GET /dashboard/api/production` — Production calendar (rolling 5-day or monthly, with calendar offset)
- `GET /dashboard/api/inventory/finished-goods` — FG inventory matching dashboard_config.json panels
- `GET /dashboard/api/inventory/batches` — Batch inventory with estimated batch counts
- `GET /dashboard/api/inventory/ingredients` — Ingredients by category per dashboard_config.json
- `GET /dashboard/api/activity/shipments` — Recent shipments with lot details
- `GET /dashboard/api/activity/receipts` — Recent receipts with lot details
- `GET /dashboard/api/lot/{lot_code}` — Lot detail with transaction timeline (color-coded by type)
- `GET /dashboard/api/search?q=...` — Global search (products, lots, orders, customers)
- `GET/POST/PUT/DELETE /dashboard/api/notes` — Notes/todos/reminders CRUD
- `PUT /dashboard/api/notes/{id}/toggle` — Toggle note done/open

### Customers

- `POST /customers` — Create customer (auto-creates during order creation if name not found)
- `GET /customers` — List all customers
- `GET /customers/search?q=...` — Search by name
- `PATCH /customers/{id}` — Update customer fields

### Other

- `GET /` — Root: returns version, features list, database status
- `GET /health` — Health check with database connectivity
- `GET /transactions/history` — Transaction history with type/product/date filters
- `GET /reason-codes` — Valid reason codes for adjustments, lot reassignment, found inventory
- `POST /admin/sql` — Read-only SQL execution (SELECT only, for diagnostics)

---

## Key Business Logic Details

### Product Types
- **ingredient** — Raw materials (oats, nuts, oils, chocolate, coconut, packaging)
- **batch** — Intermediate products created by mixing ingredients (e.g., "Batch Classic Granola #9", "Batch BS Dark Chocolate Granola 350")
- **finished_good** — Sellable products packed from batches (e.g., "Granola Classic 25 LB", "BS Granola – Dark Chocolate – 6x7 OZ Case")

### Product Resolution (3-tier)

`resolve_product_full()` is used by all transaction endpoints:
1. **Exact name match** (case-insensitive) → returns immediately
2. **Exact odoo_code match** (case-insensitive) → returns immediately
3. **Fuzzy match** (ILIKE '%term%' on name OR odoo_code) → if 1 match returns it, if multiple returns 400 with suggestions list
4. No match → 404

Returns full row: id, name, odoo_code, default_batch_lb, default_case_weight_lb, yield_multiplier

### Customer Resolution

`resolve_customer_id()` works similarly but **auto-creates** customers if no match found. This allows order creation without requiring a separate customer creation step.

### Sibling SKU Detection

`get_sibling_skus()` uses PostgreSQL array aggregation to find products with identical BOM ingredient sets:
```sql
SELECT bf.product_id, ARRAY_AGG(bf.ingredient_product_id ORDER BY bf.ingredient_product_id)
FROM batch_formulas bf
GROUP BY bf.product_id
HAVING ARRAY_AGG(...) = %s::int[]
```
When siblings exist during a make/commit, the API requires `confirmed_sku: true` to prevent packing under the wrong label (e.g., making "Batch BS Dark Chocolate" when you meant "Batch SS Chocolate Chip" — same ingredients, different brand).

### Private-Label SKU Protection

19 specific Odoo codes are flagged as private_label, plus any product named 'Batch BS %' or 'Batch Setton %':

Protected brands: Chef Quality (CQ), Sunshine (SS), Blue Stripes (BS), UNIPRO, Setton

`check_private_label_merge()` blocks adjustments where:
- Product is private_label AND
- Reason contains merge/deprecate/consolidate/migrate AND
- Quantity is negative

Returns 403 with suggestion to use 'repack' reason instead.

### Lot Code Generation

**Received goods:** `YY-MM-DD-XXXX-###` (e.g., `25-01-15-CTI-001`)
- XXXX = first 4 alpha chars of shipper name (uppercase), or shipper_code_override
- ### = auto-incrementing sequence per date+shipper

**Batch production:** `BYYMM-DD-###` (e.g., `B2501-15-001`)
- B prefix, then date, then sequence

### Confirmation Codes

Deterministic: `TXN-` + first 6 chars of SHA256(`txn-{transaction_id}-cns`).upper()
Same transaction_id always produces same code.

### Bilingual Support

All free-text fields support English + Spanish via `_es` suffix. `validate_bilingual()` rejects Spanish-only input — English is always required as the primary language.

### Timezone

Plant timezone: `America/New_York` (ET). All timestamps stored as UTC in PostgreSQL (TIMESTAMPTZ), converted to ET for display. `get_plant_now()` returns current time in plant timezone.

---

## Production Scheduling Engine (Detailed)

The scheduler generates a 7-day tactical production plan through these steps:

### 1. Build Calendar
`_build_schedule_calendar()` — Creates list of working days (skips weekends), applies Friday capacity modifier (default 0.5x).

### 2. Load Data
- `_load_line_config()` — Production lines + capacity modes from DB (JSON aggregation)
- `_load_product_line_map()` — product_id → line_code assignments
- `_load_demand()` — Open sales order lines (status IN confirmed/in_production/ready, line_status IN pending/partial) where ship date <= horizon end
- `_load_finished_inventory()` — On-hand for finished + batch products
- `_load_ingredient_inventory()` — On-hand for ingredients
- `_load_bom_structure()` — Returns 3 structures: fg_to_batch map, batch_to_ingredients map, batch_sizes dict

### 3. Simulated Allocation
`_simulated_allocation()`:
1. Groups demand by product_id, sums remaining (ordered - shipped) per product
2. For each product, allocates from finished goods inventory first
3. For finished goods with remaining need: looks up batch product via product_bom
4. Also checks batch product inventory before requiring production
5. Calculates batches needed: `math.ceil(net_need / batch_size_lb)`
6. Tracks overproduction: `(batches * batch_size) - net_need`
7. Returns list of production requirements with line_code assignments

### 4. Schedule to Days
`_schedule_runs_to_days()`:
1. Sorts requirements by earliest ship date
2. For each requirement, iterates through calendar days
3. If line already active on that day, uses remaining capacity
4. If line not active, finds best capacity mode that fits available labor
5. Activates line: deducts workers from day's labor pool
6. Assigns as many batches as fit within day capacity (adjusted by modifier)
7. Continues to next day if batches remain
8. Collects unscheduled orders with reasons

### 5. Ingredient Explosion
`_explode_ingredients()`: Multiplies each BOM ingredient quantity by scheduled batches, compares against on-hand inventory, flags shortages.

### Production Lines

| Line | Code | Products | Capacity Modes |
|------|------|----------|---------------|
| Granola Baking | granola | All granola batches | 2-worker: 9 batch/day, 3-worker: 16 batch/day |
| Coconut Sweetened | coconut | Coconut batches | Standard: 12 batch/day (2 workers) |
| Bulk Packing | bulk_pack | 25lb/10lb cases | 25lb: 4 pallet/day, 10lb: 9 pallet/day (2 workers) |
| Pouch Line | pouch | Retail pouches | Standard: 7,500 bags/day (3 workers) |

---

## Dashboard (Vanilla JS)

Single-page app (`dashboard/index.html` + `dashboard.js` + `dashboard.css`). No build step, no framework dependencies.

### Four Tabs

1. **Operations** — Production calendar (rolling 5-day or monthly view with navigation), finished goods inventory panels (collapsible with lot-level drill-down), batch inventory, ingredient inventory by category
2. **Activity** — Recent shipments and receipts with expandable transaction details. Lot codes are clickable links opening the lot detail panel.
3. **Notes** — Notes/todos/reminders with category filtering (all/note/todo/reminder), "show completed" toggle, create/edit modal with priority, due date, and entity linking (pin to product/lot/customer/supplier)
4. **Sales Orders** — Order list with status dropdown filter, customer search, overdue-only toggle. Clickable rows open order detail view with lines, shipment history, and value totals.

### Key UI Features
- **Dark/light theme** toggle (persisted to localStorage)
- **Global search** — Real-time fuzzy search across products, lots, sales orders, customers via `/dashboard/api/search`
- **Lot detail side panel** — Shows lot info + transaction timeline with color-coded events (receive=green, ship=red, make=blue, etc.)
- **Expandable rows** — Inventory tables have clickable rows showing lot-level breakdowns
- **Collapsible panels** — Inventory sections persist expanded/collapsed state via sessionStorage
- **Stock badges** — Color-coded: healthy (green), low (yellow), critical (red), zero (gray)
- **XSS protection** — All user content escaped via `escHtml()` function

### Dashboard Config (`dashboard_config.json`)

Defines the inventory panel structure:

**Finished Goods Panels:**
- 25 LB Bulk Cases (15 SKUs): Granola Classic through Setton variants
- 10 LB Cases (2 SKUs): Granola Crunchy CNS, CQ Granola
- 12x10 OZ Retail — Sunshine line (5 SKUs): SS Original, Chocolate Chip, Cranberry, Low Carb variants
- 6x7 OZ Retail — Blue Stripes line (4 SKUs): BS Dark Chocolate, PB Banana, Hazelnut Butter, Almond Butter

**Coconut Panel:** 11 SKUs — Fancy/Flake/Medium in CNS/UNIPRO/CQ brands (10lb and 25lb)

**Batch SKUs:** 18 batch products with standard batch sizes (350-451 lbs where defined)

**Ingredient Categories:**
- BS Ingredients (17 items): Almond butter, cacao nibs, coconut sugar, hazelnuts, etc.
- BS Packaging (4 items): Printed bags for each BS flavor
- Coconut Core (7 items): Desiccated coconut varieties, sugar, glycol, corn starch, salt
- Granola General (31 items): Oats, oils, nuts, chocolate chips, dried fruit, flavors, spices

### API Integration

Dashboard fetches from `/dashboard/api/*` endpoints (no auth) for local data. Sales orders are fetched from the external API base URL (`https://fastapi-production-b73a.up.railway.app`) with API key authentication.

---

## Environment & Deployment

### Environment Variables
- `DATABASE_URL` — PostgreSQL connection string (Supabase)
- `API_KEY` — API authentication key (passed as `X-API-Key` header)

### Deployment
- **API:** Railway (FastAPI + Uvicorn, auto-deploys from main branch)
- **Dashboard:** Netlify (static files from `/dashboard/`)
- Dashboard connects to API at `https://fastapi-production-b73a.up.railway.app`

### Database
- PostgreSQL on Supabase
- Migrations run automatically on startup via the `startup()` function
- `schema_migrations` table tracks applied migration versions

---

## Conventions & Patterns

1. **Single-file backend** — Everything in `main.py`. No separate route files, models files, or service layers.
2. **Raw SQL everywhere** — All database queries use raw psycopg2 with `RealDictCursor`. No ORM, no query builder.
3. **Preview/Commit pattern** — All inventory mutations go through preview (dry-run) → commit (execute). Preview never writes to DB.
4. **Append-only ledger** — Inventory is never mutated in place. Every change is a new transaction with positive/negative lines. `quantity_lb > 0` = inflow, `< 0` = outflow.
5. **FIFO allocation** — Oldest lots consumed first (ORDER BY lot.id ASC) during ship/pack/make operations.
6. **Row-level locking** — `SELECT ... FOR UPDATE` on lots before balance checks in commit endpoints to prevent race conditions.
7. **Advisory locks** — `pg_advisory_xact_lock(1)` used in receive/commit for lot code sequence generation.
8. **ThreadedConnectionPool** — psycopg2 pool (2-20 connections) with context manager auto-commit/rollback.
9. **Pydantic models** — Request validation via Pydantic BaseModel classes with custom validators.
10. **No frontend framework** — Dashboard is vanilla JS with no build step. DOM manipulation, fetch API, manual state management.
11. **Bilingual** — All user-facing text fields support English + Spanish (`_es` suffix). English required, Spanish optional.
12. **OpenAPI specs** — Both standard and GPT-optimized OpenAPI schemas maintained alongside code.
13. **Deterministic IDs** — Confirmation codes derived from transaction_id via SHA256, lot codes derived from date + shipper + sequence.
14. **CORS wide open** — `allow_origins=["*"]` for dashboard access from Netlify domain.

---

## Recent Development History

```
4a89c1b Consolidate scheduling into single POST /schedule endpoint
6f6617c Merge multi-lot FIFO into /ship/commit, remove /ship/multi-lot/* endpoints
473dcf4 Add production-scheduling to features list
5fca8a3 Add 7-day tactical production scheduling engine
91e35ea Add Sales Orders tab to factory dashboard
01d6941 Add confirmation codes to all transaction commit endpoints
f0e1042 Add lot-level breakdown to dashboard ingredient rows
c4e8f91 Fix open order filtering, migrate legacy status, flag non-weight items
4d93d8c Default new orders to confirmed status and add warnings to order list
5efaddf Add yield multiplier, lot hints, grouped ingredients, day-summary endpoint
fc65e72 Add product_bom table, FG→batch mappings, /production/requirements
085d6c8 Add water auto-exclusion and internal packing endpoints
073fad8 Add notes/to-dos/reminders feature with dashboard UI
2c3c32f v2.4.1: SKU disambiguation for make/pack transactions
3bcef8b Dark mode redesign with light/dark toggle and color-coded badges
```

---

## Pydantic Request Models Reference

```python
# Core transactions
ReceiveRequest(product_name, cases, case_size_lb, shipper_name, bol_reference, shipper_code_override?)
ShipRequest(product_name, quantity_lb, customer_name, order_reference, lot_code?)
MakeRequest(product_name, batches, lot_code?, ingredient_lot_overrides?, excluded_ingredients?, confirmed_sku?)
PackRequest(source_product, target_product, cases, case_weight_lb?, lot_allocations?, target_lot_code?)
AdjustRequest(product_name, lot_code, adjustment_lb, reason, reason_es?)

# Products
QuickCreateProductRequest(product_name, product_type, uom, storage_type, name_confidence, notes?, notes_es?)
QuickCreateBatchProductRequest(product_name, category, production_context, name_confidence, notes?, notes_es?)
ProductUpdate(default_case_weight_lb?, default_batch_lb?, yield_multiplier?, active?)
VerifyProductRequest(action, verified_name?, notes?, notes_es?)

# Inventory
LotReassignmentRequest(to_product_id, reason_code, reason_notes?, reason_notes_es?)
AddFoundInventoryRequest(product_id, quantity, uom, reason_code, found_location?, estimated_age, suspected_supplier?, suspected_bol?, notes?, notes_es?)

# Sales
CustomerCreate(name, contact_name?, email?, phone?, address?, notes?, notes_es?)
CustomerUpdate(name?, contact_name?, email?, phone?, address?, notes?, notes_es?, active?)
OrderCreate(customer_name, requested_ship_date?, lines: List[OrderLineInput], notes?, notes_es?)
OrderLineInput(product_name, quantity?, unit?, case_weight_lb?, quantity_lb?, unit_price?, notes?, notes_es?)
OrderStatusUpdate(status)
OrderHeaderUpdate(requested_ship_date?, notes?, notes_es?, customer_id?)
AddOrderLines(lines: List[OrderLineInput])
ShipOrderRequest(ship_all?, lines?: List[ShipOrderLineRequest])
ShipOrderLineRequest(line_id, quantity_lb)

# BOM
BomLineCreate(ingredient_product_id, quantity_lb, exclude_from_inventory?)
BomLineUpdate(quantity_lb?, exclude_from_inventory?)
ProductBomCreate(finished_product_id, component_product_id, quantity?, uom?)

# Notes
NoteCreate(category, title, body?, priority?, due_date?, entity_type?, entity_id?)
NoteUpdate(title?, body?, priority?, status?, due_date?, entity_type?, entity_id?)

# Admin
AdminSQLQuery(sql)  # SELECT only
LotMergeRequest(source_lot_id, target_lot_id, reason)
```

---

## Lot Traceability Fix: Canonical Lot Identity + Find-or-Create

### Principle

A physical lot must map to exactly one canonical `lot_id` in the system:
- `lots.id` (integer) = internal identity (never changes)
- `lots.lot_code` (text) = physical label (must match the floor)

The system must honor physical lot codes when provided, and generate a code only when none exists.

### Rules
1. All lot-creation endpoints implement find-or-create by `(product_id, lot_code)` via the `find_or_create_lot()` helper
2. Duplicate `(product_id, lot_code)` combinations are forbidden via unique index `lots_product_id_lot_code_key`
3. Repairs are done via controlled lot merge operations (`POST /admin/lots/merge`)
4. Lot quantities are derived from the ledger, never manually adjusted during repair
5. Merge endpoint rejects cross-product merges (`product_id` must match)

### Find-or-Create Pattern

All 5 lot-creation endpoints use the same `find_or_create_lot()` helper:
```python
def find_or_create_lot(cur, product_id, lot_code, entry_source, ...) -> (lot_id, is_new):
    INSERT INTO lots (...) VALUES (...) ON CONFLICT (product_id, lot_code) DO NOTHING;
    SELECT id FROM lots WHERE product_id = %s AND lot_code = %s;
```

### Affected Endpoints
- `/receive/commit` — Now accepts optional `lot_code` in request body
- `/make/commit` — Already accepted `lot_code`; now uses find-or-create
- `/pack/commit` — Already accepted `target_lot_code`; now uses find-or-create
- `/inventory/found` — Now accepts optional `lot_code` in request body
- `/inventory/found-with-new-product` — Now accepts optional `lot_code` in request body

### New Admin Endpoints
- `GET /admin/lots/duplicates` — Scans for duplicate `(product_id, lot_code)` pairs. Returns grouped results for review.
- `POST /admin/lots/merge` — Merges source lot into target lot:
  1. Validates both lots exist and are not already merged
  2. Validates same `product_id` (cross-product merge = 400 error)
  3. Locks both lots (`SELECT ... FOR UPDATE`)
  4. Moves all `transaction_lines` and `ingredient_lot_consumption` references
  5. Marks source lot as `status='merged'` with `merged_into_lot_id`, `merged_at`, `merge_reason`
  6. Returns computed balance from ledger

### Lots Table Merge Columns (Migration 008)
- `status` TEXT DEFAULT 'active' — 'active' or 'merged'
- `merged_into_lot_id` INTEGER FK→lots — the canonical survivor
- `merged_at` TIMESTAMPTZ — when the merge happened
- `merge_reason` TEXT — audit trail

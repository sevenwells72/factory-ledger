# Factory Ledger Dashboard Actionability Audit

**Date:** 2026-03-23
**Scope:** Backend API (Railway), Dashboard (Netlify), GPT integration

---

## 1. Complete Endpoint Inventory

### 1A. READ Endpoints (GET)

| Route | Auth | Description |
|-------|------|-------------|
| `GET /` | API Key | Root / welcome |
| `GET /health` | API Key | Health check |
| `GET /products/search` | API Key | Search products by name/code |
| `GET /products/unverified` | API Key | List unverified products |
| `GET /products/test-batches` | API Key | List test batch products |
| `GET /products/{product_id}` | API Key | Get single product detail |
| `GET /inventory/current` | API Key | Current inventory summary |
| `GET /inventory/{item_name}` | API Key | Inventory for specific item |
| `GET /inventory/found/queue` | API Key | Found-inventory review queue |
| `GET /lots/by-supplier-lot/{supplier_lot_code}` | API Key | Lookup lots by supplier lot code |
| `GET /lots/by-code/{lot_code}` | API Key | Lookup lot by internal lot code |
| `GET /lots/{lot_id}` | API Key | Get lot by ID |
| `GET /trace/batch/{lot_code}` | API Key | Forward trace: batch → finished goods |
| `GET /trace/ingredient/{lot_code}` | API Key | Forward trace: ingredient → batches → FG |
| `GET /transactions/history` | API Key | Transaction history with filters |
| `GET /bom/products` | API Key | List products with BOMs |
| `GET /bom/batches/{batch_id}/formula` | API Key | Get batch formula/recipe |
| `GET /reason-codes` | API Key | List adjustment reason codes |
| `GET /customers` | API Key | List all active customers |
| `GET /customers/search` | API Key | Search customers by name/alias |
| `GET /sales/orders` | API Key | List sales orders (with filters) |
| `GET /sales/orders/fulfillment-check` | API Key | Check fulfillment feasibility |
| `GET /sales/orders/{order_id}` | API Key | Get full order detail |
| `GET /sales/orders/{order_id}/packing-slip` | Flexible* | Generate packing slip (HTML) |
| `GET /sales/dashboard` | API Key | Sales dashboard summary |
| `GET /production/requirements` | API Key | Production requirements/planning |
| `GET /production/day-summary` | API Key | Daily production summary |
| `GET /audit/integrity` | **None** | System integrity score |
| `GET /admin/bom/search` | API Key | Search products for BOM editing |
| `GET /admin/bom/{product_id}/lines` | API Key | Get BOM lines for product |
| `GET /admin/product-bom` | API Key | Get product-BOM mappings |
| `GET /admin/lots/duplicates` | API Key | Scan for duplicate lots |
| **Dashboard API (no auth):** | | |
| `GET /dashboard/inventory` | API Key | Legacy dashboard inventory |
| `GET /dashboard/low-stock` | API Key | Legacy low-stock alerts |
| `GET /dashboard/today` | API Key | Legacy today summary |
| `GET /dashboard/lots` | API Key | Legacy lots summary |
| `GET /dashboard/production` | API Key | Legacy production summary |
| `GET /dashboard/api/production` | **None** | Production calendar data |
| `GET /dashboard/api/inventory/finished-goods` | **None** | Finished goods on-hand |
| `GET /dashboard/api/inventory/batches` | **None** | Batch inventory on-hand |
| `GET /dashboard/api/inventory/ingredients` | **None** | Ingredient inventory on-hand |
| `GET /dashboard/api/activity/shipments` | **None** | Recent shipments log |
| `GET /dashboard/api/activity/receipts` | **None** | Recent receipts log |
| `GET /dashboard/api/lot/{lot_code}` | **None** | Lot detail (for side panel) |
| `GET /dashboard/api/product/{product_id}/lots` | **None** | Product lots listing |
| `GET /dashboard/api/search` | **None** | Global search |
| `GET /dashboard/api/notes` | **None** | List notes/todos |

*Flexible = accepts API key via header OR query param (for browser access)

### 1B. WRITE Endpoints (POST/PUT/PATCH/DELETE)

| Route | Method | Auth | Description | Key Parameters |
|-------|--------|------|-------------|----------------|
| `POST /receive` | POST | API Key | Receive inventory (preview/commit) | `product_name`, `cases`, `case_size_lb`, `shipper_name`, `bol_reference`, `supplier_lot_code`, `mode` |
| `POST /ship` | POST | API Key | Standalone ship (preview/commit) | `product_name`, `quantity_lb`, `customer_name`, `order_reference`, `lot_code`, `mode` |
| `POST /make` | POST | API Key | Production run (preview/commit) | `product_name`, `batches`, `lot_code`, `ingredient_lot_overrides`, `mode` |
| `POST /pack` | POST | API Key | Pack batch→FG cases (preview/commit) | `source_product`, `target_product`, `cases`, `case_weight_lb`, `mode` |
| `POST /adjust` | POST | API Key | Inventory adjustment (preview/commit) | `product_name`, `lot_code`, `adjustment_lb`, `reason`, `mode` |
| `POST /void/{transaction_id}` | POST | API Key | Void a transaction | `transaction_id` (path) |
| `PATCH /lots/{lot_code}/supplier-lot` | PATCH | API Key | Set supplier lot code on lot | `supplier_lot_code` |
| `POST /products/quick-create` | POST | API Key | Create a new product | `product_name`, `product_type`, `uom` |
| `POST /products/quick-create-batch` | POST | API Key | Create a new batch product | `product_name`, `category`, `production_context` |
| `POST /products/resolve` | POST | API Key | Bulk resolve product names | `names[]` |
| `POST /products/{product_id}/verify` | POST | API Key | Verify/rename unverified product | `action`, `verified_name` |
| `POST /lots/{lot_id}/reassign` | POST | API Key | Reassign lot to different product | `to_product_id`, `reason_code` |
| `POST /inventory/found` | POST | API Key | Log found inventory (existing product) | `product_id`, `quantity`, `reason_code` |
| `POST /inventory/found-with-new-product` | POST | API Key | Log found inventory (new product) | `product_name`, `product_type`, `quantity`, `reason_code` |
| `POST /customers` | POST | API Key | Create customer | `name`, `contact_name`, `email`, `phone` |
| `PATCH /customers/{customer_id}` | PATCH | API Key | Update customer | `name`, `contact_name`, `email`, `phone`, `aliases[]` |
| `POST /sales/orders` | POST | API Key | Create sales order | `customer_name`, `requested_ship_date`, `lines[]` |
| `PATCH /sales/orders/{order_id}/status` | PATCH | API Key | Update order status | `status` (state machine enforced) |
| `PATCH /sales/orders/{order_id}` | PATCH | API Key | Update order header | `requested_ship_date`, `notes`, `customer_name` |
| `POST /sales/orders/{order_id}/lines` | POST | API Key | Add line to order | `product_name`, `quantity_lb` |
| `PATCH /sales/orders/{order_id}/lines/{line_id}/cancel` | PATCH | API Key | Cancel order line | (no body needed) |
| `PATCH /sales/orders/{order_id}/lines/{line_id}/update` | PATCH | API Key | Update order line quantity | `quantity_lb`, `unit`, `case_weight_lb` |
| `POST /sales/orders/{order_id}/ship` | POST | API Key | Ship against sales order (preview/commit) | `mode`, `ship_all`, `lines[]` |
| `PUT /admin/products/{product_id}` | PUT | API Key | Update product fields | `odoo_code`, `case_size_lb`, `default_batch_lb`, etc. |
| `POST /admin/bom/{product_id}/lines` | POST | API Key | Add BOM line | `ingredient_product_id`, `quantity_lb` |
| `PUT /admin/bom/lines/{line_id}` | PUT | API Key | Update BOM line | `quantity_lb`, `exclude_from_inventory` |
| `DELETE /admin/bom/lines/{line_id}` | DELETE | API Key | Delete BOM line | (path param only) |
| `POST /admin/product-bom` | POST | API Key | Create product-BOM mapping | |
| `DELETE /admin/product-bom/{mapping_id}` | DELETE | API Key | Delete product-BOM mapping | |
| `POST /admin/sql` | POST | API Key | Run read-only SQL (SELECT only) | `sql` |
| `POST /admin/lots/merge` | POST | API Key | Merge duplicate lots | `source_lot_id`, `target_lot_id`, `reason` |
| `POST /schedule` | POST | API Key | Create/update production schedule | |
| **Dashboard API (no auth):** | | | | |
| `POST /dashboard/api/notes` | POST | **None** | Create note/todo | `category`, `title`, `body`, `priority`, `due_date` |
| `PUT /dashboard/api/notes/{note_id}` | PUT | **None** | Update note | (partial update fields) |
| `DELETE /dashboard/api/notes/{note_id}` | DELETE | **None** | Delete note | (path param only) |
| `PUT /dashboard/api/notes/{note_id}/toggle` | PUT | **None** | Toggle note open/done | (path param only) |

---

## 2. Dashboard Page Inventory

| Page / Tab | File | Data Displayed | API Endpoints Called | Current Actions | Proposed Actions |
|------------|------|---------------|---------------------|-----------------|-----------------|
| **Operations → Production Calendar** | `index.html` (tab) | Weekly/monthly production runs | `GET /dashboard/api/production` | View only | Schedule production, log a make run |
| **Operations → Finished Goods** | `index.html` (tab) | On-hand cases/lbs by SKU, lot details | `GET /dashboard/api/inventory/finished-goods` | Click lot → side panel | Adjust inventory, quick ship |
| **Operations → Batch Inventory** | `index.html` (tab) | Batch product on-hand with bar charts | `GET /dashboard/api/inventory/batches` | Click lot → side panel | Pack batch into cases |
| **Operations → Ingredients** | `index.html` (tab) | Ingredient on-hand by category | `GET /dashboard/api/inventory/ingredients` | Click lot → side panel | Receive inventory, adjust |
| **Activity → Shipping** | `index.html` (tab) | Recent shipment transactions | `GET /dashboard/api/activity/shipments` | View only | Link to order, void shipment |
| **Activity → Receiving** | `index.html` (tab) | Recent receipt transactions | `GET /dashboard/api/activity/receipts` | View only | Void receipt |
| **Notes** | `index.html` (tab) | Notes/todos/reminders with filters | `GET/POST/PUT/DELETE /dashboard/api/notes/*` | **Full CRUD** (already actionable) | — (done) |
| **Sales Orders → List** | `index.html` (tab) | Orders table with status/customer filters | `GET /sales/orders` (with API key) | View, filter, click to detail | Create new order |
| **Sales Orders → Detail** | `index.html` (tab) | Order header, KPIs, line items | `GET /sales/orders/{id}` (with API key) | View only | **Ship order**, update status, edit header, add/cancel lines |
| **Material Flow** | `sankey.html` | Sankey diagram of material flow | `GET /dashboard/api/*` (various) | View only | — (visualization, no actions needed) |
| **Production Lines** | `process-flow.html` | Process flow diagrams | `GET /dashboard/api/*` (various) | View only | — (visualization, no actions needed) |
| **Traceability** | `traceability.html` | Forward/backward lot trace | `GET /trace/*` endpoints | View only | — (read-only by nature) |
| **Lot Detail Panel** | `index.html` (overlay) | Lot transactions, supplier lot, dates | `GET /dashboard/api/lot/{lot_code}` | View only | Adjust lot, set supplier lot code |
| **Global Search** | `index.html` (header) | Products, lots, orders, customers | `GET /dashboard/api/search` | Navigate to results | — |
| **Health Badge** | `index.html` (header) | Integrity score | `GET /audit/integrity` | View only | — |

---

## 3. Gap Analysis

### 3A. Write Endpoints → Dashboard Page Mapping

| Write Operation | Endpoint Exists? | Natural Dashboard Location | Implementation Complexity |
|----------------|-----------------|---------------------------|--------------------------|
| **Ship order** | `POST /sales/orders/{id}/ship` | Sales Orders → Detail page | Medium — preview/confirm flow |
| **Update order status** | `PATCH /sales/orders/{id}/status` | Sales Orders → Detail page | Low — dropdown with valid transitions |
| **Receive inventory** | `POST /receive` | Operations → Ingredients, or new Receiving tab | Medium — multi-field form |
| **Adjust inventory** | `POST /adjust` | Lot Detail Panel or FG/Batch/Ingredient cards | Low — simple form (lot, lbs, reason) |
| **Create sales order** | `POST /sales/orders` | Sales Orders → List (+ New Order button) | High — dynamic line items, customer search |
| **Edit order header** | `PATCH /sales/orders/{id}` | Sales Orders → Detail page | Low — inline edit of date/notes |
| **Add order line** | `POST /sales/orders/{id}/lines` | Sales Orders → Detail page | Medium — product search + quantity |
| **Cancel order line** | `PATCH /sales/orders/{id}/lines/{line_id}/cancel` | Sales Orders → Detail page | Low — button with confirm |
| **Make (production run)** | `POST /make` | Operations → Production Calendar | High — BOM deductions, lot overrides |
| **Pack** | `POST /pack` | Operations → Batch Inventory | Medium — source/target product, cases |
| **Void transaction** | `POST /void/{id}` | Activity → Shipping/Receiving rows | Low — button with confirm |
| **Create customer** | `POST /customers` | Sales Orders → New Order flow | Medium — form with optional fields |
| **Update supplier lot** | `PATCH /lots/{lot_code}/supplier-lot` | Lot Detail Panel | Low — inline edit |

### 3B. GPT Multi-Step Operations Without Direct Endpoint

| GPT Workflow | Steps Involved | Dashboard Solution |
|-------------|---------------|-------------------|
| "Receive 50 cases of Granola Classic 25 LB from ACME" | GPT resolves product name → calls `POST /receive` preview → shows user → calls commit | Dashboard form with product dropdown/search, auto-preview, confirm button |
| "Ship SO260312005" | GPT looks up order → calls `POST /sales/orders/{id}/ship` preview → reviews inventory → calls commit | Dashboard already has order detail; add "Ship" button → preview modal → confirm |
| "Make 2 batches of Classic Granola #9" | GPT resolves product → calls `POST /make` preview → reviews BOM/lots → handles lot overrides → commits | Complex form; start with simple "make with defaults" then iterate |
| "Adjust lot ABC-123 by -50 lb, reason: recount" | GPT resolves product from lot → calls `POST /adjust` | Simple form in lot detail panel |

**Key finding:** All GPT workflows ultimately call a single endpoint — the GPT's "multi-step" is just product name resolution + preview + commit. The preview/commit pattern in the API is already dashboard-friendly.

### 3C. Validation Gaps for Form-Based UI

| Endpoint | Issue | Recommendation |
|----------|-------|----------------|
| `POST /receive` | Returns 500 with generic error string on some failures | Standardize all errors to `{error: string, field?: string}` |
| `POST /make` | Complex sibling SKU confirmation flow (`confirmed_sku` field) | Needs a "confirm sibling" modal in dashboard |
| `POST /sales/orders` | Unit defaults to `lb` if not specified (silent) | Dashboard form should always require explicit unit selection |
| `POST /ship` (standalone) | `force_standalone` and `force_create_customer` flags | Dashboard should show warning modals instead of force flags |
| General | Some 500 errors return `{"error": str(e)}` — raw Python exceptions | Wrap all in user-friendly messages |

---

## 4. Authentication & Security Assessment

### Current Auth Mechanism

- **Method:** Single shared API key via `X-API-Key` header
- **Key value:** Hardcoded in dashboard JS: `ledger-secret-2026-factory`
- **Two tiers:**
  - `/dashboard/api/*` endpoints: **No auth** (designed for the Netlify dashboard)
  - All other endpoints: Require API key via `verify_api_key` dependency
  - Packing slip: `verify_api_key_flexible` — accepts key via header OR `?key=` query param

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ← Wide open
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**CORS is fully permissive.** The Netlify dashboard can already make POST/PATCH/DELETE requests to the Railway API. No CORS changes needed for actionability.

### Security Assessment for Dashboard Writes

| Concern | Current State | Risk Level | Recommendation |
|---------|--------------|------------|----------------|
| **API key exposure** | Key is hardcoded in `dashboard.js` (client-side, visible in source) | **High** | The key is already exposed. For dashboard writes, this is acceptable if the dashboard is internal-only. If public-facing, add user auth. |
| **No user-level auth** | Single shared key, no user identity tracking | **Medium** | For audit trail, add `performed_by` field to dashboard write calls. No need for full user auth if team is small. |
| **CORS wide open** | `allow_origins=["*"]` | **Low** | Acceptable for an internal tool. Tighten to `["https://cns-factory-ledger.netlify.app"]` if desired. |
| **No rate limiting** | None | **Low** | Internal tool, low risk. Add if exposed publicly. |
| **Dashboard API unauthed** | `/dashboard/api/*` has no auth, including notes CRUD | **Medium** | Notes are already writeable without auth. Adding more writes at `/dashboard/api/*` maintains consistency but increases surface. Consider requiring API key for all dashboard writes. |

### Recommendation

For the immediate term (small internal team):
1. **Keep the shared API key approach** — it's already in the dashboard JS
2. **Add `performed_by` to all dashboard write calls** (e.g., "dashboard-user") for audit trail
3. **Tighten CORS** to Netlify domain only
4. **Add confirmation dialogs** for destructive actions (void, cancel)

---

## 5. Prioritized Implementation Plan

### Priority Ordering Criteria
- **Frequency:** How often does the team do this? (daily > weekly > occasional)
- **Ease:** How close is the existing endpoint to being dashboard-ready?
- **Risk:** What's the blast radius of a mistake?

### Priority Ranking

| # | Action | Frequency | Ease | Risk | Priority Score |
|---|--------|-----------|------|------|---------------|
| 1 | Ship a sales order | Daily | Medium | Medium | **Highest** |
| 2 | Update order status | Daily | Easy | Low | **High** |
| 3 | Adjust inventory (lot) | Daily | Easy | Low | **High** |
| 4 | Receive inventory | Daily | Medium | Low | **High** |
| 5 | Edit order header (ship date, notes) | Weekly | Easy | Low | **Medium** |
| 6 | Void a transaction | Weekly | Easy | High | **Medium** |
| 7 | Create sales order | Weekly | Hard | Medium | **Medium** |
| 8 | Cancel order line | Weekly | Easy | Medium | **Medium** |
| 9 | Pack batch → FG | Daily | Medium | Medium | **Medium** |
| 10 | Make (production run) | Daily | Hard | High | **Lower** (complex BOM) |
| 11 | Set supplier lot code | Occasional | Easy | Low | **Lower** |
| 12 | Create customer | Occasional | Easy | Low | **Lower** |

---

### Top 5 Action Sketches

#### Action 1: Ship a Sales Order

**Dashboard page:** Sales Orders → Order Detail view
**UI interaction:**
- Add a **"Ship Order"** button in the order detail header (visible when status is `confirmed`, `in_production`, `ready`, or `partial_ship`)
- Clicking opens a **modal** with:
  - Preview table: each unfulfilled line showing product, ordered, already shipped, remaining, on-hand, can-ship
  - Warnings highlighted (e.g., insufficient inventory)
  - Option to ship all or edit per-line quantities
  - **"Confirm Ship"** button (green) and **"Cancel"** button
- On confirm: calls `POST /sales/orders/{id}/ship` with `mode=commit`
- On success: refresh order detail, show success toast

**Endpoints called:**
1. `POST /sales/orders/{id}/ship` with `mode=preview` (on button click)
2. `POST /sales/orders/{id}/ship` with `mode=commit` (on confirm)

**Confirmation/error handling:**
- Preview step IS the confirmation (shows exactly what will happen)
- Warnings for short inventory shown in orange
- Error toast if commit fails
- Disable button while request is in-flight

```
┌─────────────────────────────────────────────┐
│  Ship Order SO260312005                  ✕  │
├─────────────────────────────────────────────┤
│  Product          Remaining  On-Hand  Ship  │
│  ─────────────────────────────────────────  │
│  Granola Classic   500 lb    620 lb   500   │
│  Granola Choc      250 lb    180 lb   180   │
│                              ⚠ Short 70 lb  │
│                                             │
│  [  Cancel  ]              [ Confirm Ship ] │
└─────────────────────────────────────────────┘
```

---

#### Action 2: Update Order Status

**Dashboard page:** Sales Orders → Order Detail view
**UI interaction:**
- Add a **status dropdown** next to the current status badge in the order detail header
- Dropdown only shows valid transitions from current state (using `MANUAL_TRANSITIONS` map)
- Selecting a new status shows an **inline confirm**: "Change status to In Production?"
- On confirm: calls `PATCH /sales/orders/{id}/status`
- On success: update badge, show toast

**Endpoints called:**
1. `PATCH /sales/orders/{order_id}/status` with `{status: "new_status"}`

**Confirmation/error handling:**
- Dropdown is pre-filtered to valid transitions (no invalid options shown)
- Inline confirm with the transition description
- "Cancelled" shows a red-highlighted confirm ("This cannot be undone")
- Error toast with backend message on failure

```
  SO260312005  [Confirmed ▾]
                ┌──────────────┐
                │ In Production│
                │ Cancelled ⚠  │
                └──────────────┘
```

---

#### Action 3: Adjust Inventory

**Dashboard page:** Lot Detail side panel (accessible from FG, Batch, or Ingredient cards)
**UI interaction:**
- Add an **"Adjust"** button at the bottom of the lot detail panel
- Clicking reveals an inline form:
  - `Adjustment (lb)`: number input (positive = add, negative = remove)
  - `Reason`: dropdown of reason codes (fetched from `GET /reason-codes`)
  - `Notes` (optional): text input
  - **"Preview"** button → shows what will happen
  - **"Confirm Adjust"** button
- On success: refresh lot detail panel and parent inventory view

**Endpoints called:**
1. `GET /reason-codes` (once, on panel open)
2. `POST /adjust` with `mode=preview`
3. `POST /adjust` with `mode=commit`

**Confirmation/error handling:**
- Preview step shows current balance, adjustment, and new balance
- Negative adjustments that would go below zero show error from backend
- Success toast with new balance

```
┌─ Lot Detail: CNS-260315-GRAN ──────────┐
│  Product: Granola Classic 25 LB         │
│  On-Hand: 1,250 lb                      │
│  ...                                    │
│                                         │
│  ┌─ Adjust Inventory ────────────────┐  │
│  │ Adjustment: [-50    ] lb          │  │
│  │ Reason:     [Recount         ▾]   │  │
│  │ Notes:      [Physical count was ] │  │
│  │                                   │  │
│  │ Preview: 1,250 → 1,200 lb        │  │
│  │ [Cancel]         [Confirm Adjust] │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

#### Action 4: Receive Inventory

**Dashboard page:** Operations tab → new **"+ Receive"** button in the header area, or a dedicated Receiving section
**UI interaction:**
- **"+ Receive"** button opens a **modal form** with:
  - `Product`: searchable dropdown (calls `GET /products/search`)
  - `Cases`: number input
  - `Case Size (lb)`: auto-populated from product's `case_size_lb`, editable
  - `Shipper/Supplier`: text input
  - `BOL Reference`: text input
  - `Supplier Lot Code`: text input (required)
  - **"Preview"** button → calls preview mode, shows summary
  - **"Confirm Receive"** button
- On success: toast, refresh finished goods / ingredients as appropriate

**Endpoints called:**
1. `GET /products/search?q=...` (as-you-type)
2. `POST /receive` with `mode=preview`
3. `POST /receive` with `mode=commit`

**Confirmation/error handling:**
- Preview shows: product name, total lbs, lot code (auto-generated or specified), lot-exists warning
- Required field validation (supplier lot code)
- Error messages from backend displayed inline

```
┌─────────────────────────────────────────┐
│  Receive Inventory                   ✕  │
├─────────────────────────────────────────┤
│  Product:    [Granola Classic 25 LB  ▾] │
│  Cases:      [20        ]               │
│  Case Size:  [25        ] lb            │
│  Shipper:    [ACME Foods           ]    │
│  BOL:        [BOL-2026-0312        ]    │
│  Supplier Lot: [SL-ABC-123         ]    │
│                                         │
│  ── Preview ──                          │
│  500 lb → lot CNS-260323-GRAN (new)     │
│                                         │
│  [Cancel]           [Confirm Receive]   │
└─────────────────────────────────────────┘
```

---

#### Action 5: Edit Order Header

**Dashboard page:** Sales Orders → Order Detail view
**UI interaction:**
- **Inline edit** on the order detail header:
  - Click the ship date → date picker appears
  - Click notes → textarea appears
  - Each shows a **"Save"** / **"Cancel"** button pair
- Only enabled when order status is `new` or `confirmed`
- On save: calls `PATCH /sales/orders/{id}` with changed fields

**Endpoints called:**
1. `PATCH /sales/orders/{order_id}` with `{requested_ship_date?, notes?}`

**Confirmation/error handling:**
- Inline save — no modal needed (low-risk edit)
- Disabled with tooltip when status is past `confirmed`
- Error toast if backend rejects (e.g., status not editable)
- Success: update displayed values in-place

```
  Order Date: 03/12/26
  Ship By:    [03/25/26 📅]  [Save] [Cancel]
  Notes:      [Rush order - customer needs by Friday  ]
              [Save] [Cancel]
```

---

## 6. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days each)
1. **Update order status** — dropdown on order detail page
2. **Edit order header** — inline date/notes editing
3. **Cancel order line** — button with confirm on order detail

### Phase 2: Core Actions (2-3 days each)
4. **Ship order** — preview/confirm modal on order detail
5. **Adjust inventory** — inline form in lot detail panel
6. **Void transaction** — button on activity log rows

### Phase 3: Forms (3-5 days each)
7. **Receive inventory** — modal form with product search
8. **Create sales order** — full form with dynamic lines
9. **Pack batch** — modal from batch inventory cards

### Phase 4: Advanced (5+ days)
10. **Make (production run)** — complex form with BOM preview, lot overrides
11. **Admin: BOM editing** — dedicated admin page

### Technical Prerequisites
- [ ] Add API key to all dashboard write calls (use existing `SALES_API_KEY` pattern)
- [ ] Create reusable UI components: confirm modal, toast notifications, inline edit, searchable dropdown
- [ ] Add `performed_by: "dashboard"` to all write requests for audit trail
- [ ] Consider tightening CORS to `["https://cns-factory-ledger.netlify.app"]`

---

## Appendix: Dashboard JS API Patterns (Current)

**Read endpoints (no auth):**
```javascript
const API_BASE = 'https://fastapi-production-b73a.up.railway.app/dashboard/api';
const res = await fetch(API_BASE + path);
```

**Sales endpoints (with auth):**
```javascript
const SALES_API_BASE = 'https://fastapi-production-b73a.up.railway.app';
const SALES_API_KEY = 'ledger-secret-2026-factory';
const res = await fetch(SALES_API_BASE + path, {
  headers: { 'X-API-Key': SALES_API_KEY }
});
```

**Notes endpoints (no auth, with body):**
```javascript
await fetch(API_BASE + '/notes', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});
```

All new dashboard write endpoints should follow the **Sales API pattern** (with API key header).

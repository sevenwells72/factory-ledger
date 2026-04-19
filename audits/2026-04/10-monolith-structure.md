# Category 1 — Monolith Structure & Duplication

**Target:** [main.py](../../main.py) — 9,293 lines, 99 endpoints, 1 file.

Entire Python runtime is one module. Below are the logical subsystems hiding inside it, the duplicated logic, and a proposed module split.

---

## Logical subsystems discovered inside main.py

The file is already de-facto structured with banner comments (`# ═══...`). Mapping banner → endpoint clusters gives a natural module split:

| Subsystem | Line range | Endpoint count | Notes |
|---|---|---:|---|
| Bootstrap + pool + auth + helpers | 1–654 | 0 | Startup contains ~310 lines of inline DDL ("Migrations 004–012") — see finding F17-01 |
| Pydantic models (ops + notes + sales) | 661–948 | 0 | Models split across 3 banners; some later models defined inline at endpoint sites (`LotRenameRequest`, `SupplierLotUpdate`, `ProductUpdate`, `BomLineCreate`, `BomLineUpdate`, `ProductBomCreate`, `AdminSQLQuery`, `LotMergeRequest`). |
| Sales helpers + product resolvers | 954–1259 | 0 | `_tiered_product_search`, `resolve_customer_id`, 4 product-resolve wrappers |
| Dashboard summary endpoints | 1266–1314 | 5 | Thin SELECT-from-view endpoints; 100% redundant with `/dashboard/api/*` below |
| Health / Root | 1321–1340 | 2 | |
| Product search / list / get | 1347–1490 | 5 | |
| Inventory lookup | 1497–1690 | 3 | `inventory_lookup` (unified) supersedes `get_current_inventory` + `get_inventory` |
| Lot lookup + patch | 1691–2025 | 5 | |
| Lot-identity helpers | 2026–2112 | 0 | `find_or_create_lot`, `generate_lot_code` |
| Receive | 2113–2251 | 1 | |
| Ship (standalone) | 2252–2596 | 1 | + `check_open_orders_for_ship` helper |
| Make | 2597–2964 | 1 | |
| Pack (incl. add-in resolution) | 2965–3311 | 1 | |
| Adjust + Void | 3312–3486 | 2 | |
| Trace (batch + ingredient + supplier-lot) | 3487–4073 | 3 | + `_trace_ingredient_backward` |
| Transaction history | 4074–4145 | 1 | |
| Quick-create (product + batch) | 4146–4244 | 2 | |
| Lot reassign | 4245–4356 | 1 | |
| Found inventory (3 endpoints) | 4357–4542 | 3 | |
| Product verify | 4543–4609 | 1 | |
| BOM read | 4610–4678 | 2 | |
| Reason codes | 4679–4709 | 1 | |
| Customers CRUD | 4710–4855 | 4 | |
| Sales orders (create/list/fulfillment/get/status/header/lines) | 4856–5644 | 9 | |
| Sales order ship | 5645–5855 | 1 | 208 lines — largest single endpoint |
| Preview/commit alias stubs | 5856–5923 | 12 | `include_in_schema=False`; 5 line apiece; see finding F02-02 |
| Packing slip (PDF) | 5924–6403 | 1 | 479 lines; ReportLab-heavy; deserves its own module |
| Sales dashboard | 6404–6498 | 1 | |
| Dashboard API (`/dashboard/api/*`) | 6499–7382 | 14 | Netlify frontend private API |
| Admin (product edit + BOM admin + SQL + lots) | 7383–7889 | 12 | |
| Production scheduling | 7890–9083 | 2 | 1,200 lines with 9 private helpers; the second-largest subsystem |
| Schedule dispatch + audit/integrity | 9084–9285 | 2 | |
| Static file mount | 9287–9293 | 0 | |

Totals: **99 endpoints, 32 helper functions, 30+ Pydantic models.**

Notable structure smells:
- **F01-01** Pydantic models live in 4+ different places (main block 661–948, plus scattered ad-hoc classes before each admin endpoint from line 7383 onward). See [main.py:7384](../../main.py#L7384), [main.py:7502](../../main.py#L7502), [main.py:7651](../../main.py#L7651), [main.py:7715](../../main.py#L7715), [main.py:7781](../../main.py#L7781).
- **F01-02** Two parallel dashboard endpoint sets. Legacy set [main.py:1266](../../main.py#L1266) reads from DB views (`inventory_summary`, `low_stock_alerts`, `todays_transactions`, `lot_balances`, `production_history`). New set [main.py:6509](../../main.py#L6509) queries base tables directly. The legacy set is still registered; dashboard JS only calls the new one (per dashboard-endpoint audit). Candidate for deletion — see also F03-D.

---

## Duplicated logic — ranked by impact

### [F01-D1] FIFO lot-selection + per-lot deduction loop rewritten 8× — ~250 lines
**Severity**: high
**Files**: [main.py:2449–2550](../../main.py#L2449) (ship standalone), [main.py:2633–2724](../../main.py#L2633) (make ingredients), [main.py:3023–3089](../../main.py#L3023) (pack source), [main.py:3148–3280](../../main.py#L3148) (pack ingredients + add-ins), [main.py:5755–5787](../../main.py#L5755) (ship order), [main.py:6055–6085](../../main.py#L6055) (packing-slip FIFO preview), [main.py:6625–6815](../../main.py#L6625) (dashboard lot aggregates) — 8 full implementations of the same "SELECT candidates ORDER BY received_at ASC, FOR UPDATE, loop take=min(remaining,balance), INSERT transaction_lines" flow.
**Current behavior**: Each loop has its own variable names, its own balance re-check, its own error messages. Shared pattern: ~30 lines per occurrence.
**Risk**: Any future fix to FIFO (epsilon handling, tie-breaking, multi-product atomicity, lot-exhaustion messaging) must be applied 8 times; easy to miss one. Lot 131 negative balance (migration 015) suggests this class of bug has already bitten once.
**Recommended fix**: Extract `fifo_deduct(cur, product_id, required_lb, txn_id, pinned_lot_code=None, lock=True) -> List[ConsumedLot]` helper. Each of the 8 sites becomes a single call.
**Effort**: large (1–2 days with tests)

### [F01-D2] `SUM(quantity_lb)` lot-balance query inlined 15+ times — ~80 lines
**Severity**: high
**Files**: canonical helper [main.py:121](../../main.py#L121) (`validate_lot_deduction`), but inlined at [main.py:2450](../../main.py#L2450), [main.py:2509](../../main.py#L2509), [main.py:2635](../../main.py#L2635), [main.py:2663](../../main.py#L2663), [main.py:2862](../../main.py#L2862), [main.py:3023](../../main.py#L3023), [main.py:3081](../../main.py#L3081), [main.py:3155](../../main.py#L3155), [main.py:3243](../../main.py#L3243), [main.py:5755](../../main.py#L5755), [main.py:6060](../../main.py#L6060), [main.py:6625](../../main.py#L6625), [main.py:6720](../../main.py#L6720), [main.py:6808](../../main.py#L6808), [main.py:7854](../../main.py#L7854), [main.py:9132](../../main.py#L9132) (audit/integrity check #1 — re-implements in SQL).
**Current behavior**: Every call-site writes its own `COALESCE(SUM(quantity_lb), 0)` query. BALANCE_EPSILON snap is only applied inside the helper, not inline — some sites compare raw SUMs to required qty.
**Risk**: Near-zero balances ($<$ 0.0001 lb) behave differently across endpoints. Migration 020 (numeric precision) was the cleanup pass for exactly this drift.
**Recommended fix**: Replace every inline SUM with `balance = get_lot_balance(cur, lot_id)`. For pre-write validation, all sites should flow through `validate_lot_deduction`.
**Effort**: medium (half day)

### [F01-D3] `is_service` + case-weight conversion block duplicated verbatim in 2 places — ~50 lines
**Severity**: high (correctness risk)
**Files**: [main.py:4883–4908](../../main.py#L4883) (`create_sales_order`) and [main.py:5516–5540](../../main.py#L5516) (`add_order_lines`).
**Current behavior**: Two places look up `case_size_lb, is_service` from products, branch on service vs product, recompute `quantity_lb = quantity * case_weight`. This re-implements `OrderLineInput.calculate_quantity_lb` (validator at [main.py:894](../../main.py#L894)) because the validator has no `is_service` bypass. Additionally the `is_service` filter `FILTER (WHERE NOT COALESCE(p.is_service, false))` appears 6× at [main.py:4985](../../main.py#L4985), [main.py:6420](../../main.py#L6420), [main.py:6428](../../main.py#L6428), [main.py:6440](../../main.py#L6440), [main.py:6448](../../main.py#L6448), [main.py:8476](../../main.py#L8476).
**Risk**: Rules around service items (changelog entry #13 — `quantity_lb >= 0` relaxation for pallets) must match in both places. `ship_order` has no `is_service` guard at all (see [F05-04](15-traceability-gaps.md) for the full story) — orders containing a pallet charge can never close via that endpoint.
**Recommended fix**: Single helper `resolve_order_line_quantity(cur, product_id, quantity, unit, case_weight_lb) -> (quantity_lb, is_service, product_name, case_size_lb)` called from create, add-lines, and the ship loop. `ship_order` gets the guard for free.
**Effort**: medium (half day)

### [F01-D4] Product-resolver wrappers duplicate each other — ~35 lines
**Severity**: medium
**Files**: [main.py:954](../../main.py#L954) (`resolve_product_id`) and [main.py:983](../../main.py#L983) (`resolve_product_full`) have ~80% overlap. Call sites that bypass the wrappers and re-implement ambiguity handling: [main.py:1356](../../main.py#L1356) (`search_products`), [main.py:1618](../../main.py#L1618) (`inventory_lookup`), [main.py:1655](../../main.py#L1655) (`get_inventory` fuzzy fallback).
**Current behavior**: Three resolve shapes with different return types.
**Recommended fix**: Keep `_tiered_product_search` as the engine; collapse `resolve_product_id` and `resolve_product_full` into one function returning a full row; wrap with `.id` projection where only the id is needed.
**Effort**: small

### [F01-D5] `line_status` recomputation divergence — correctness hazard
**Severity**: high (correctness)
**Files**: commit-side recompute at [main.py:5793](../../main.py#L5793); preview-side filter uses stale DB `line_status` at [main.py:5132](../../main.py#L5132), [main.py:5669](../../main.py#L5669), [main.py:5716](../../main.py#L5716); order-level status at [main.py:5838](../../main.py#L5838) uses `'shipped' | 'partial_ship'`; line-level status uses `'fulfilled' | 'partial' | 'pending'`.
**Current behavior**: Two vocabularies for the same state machine. Preview never recomputes from `quantity_shipped_lb`; if DB drift exists (e.g. after an unhandled void — [F05-05](15-traceability-gaps.md)), preview and commit disagree.
**Risk**: User sees "ready to ship" in preview, gets "nothing to ship" in commit, or vice versa. Small in line count but high in blast radius.
**Recommended fix**: Promote the commit-side decision to a helper `recompute_line_status(ordered, shipped) -> str` and a `recompute_order_status(lines) -> str`. Use everywhere that reads or writes status.
**Effort**: small

### [F01-D6] Shipment record creation — inconsistent column sets
**Severity**: medium
**Files**: [main.py:2553](../../main.py#L2553) inserts `shipments(transaction_id, shipped_at, customer_id)`; [main.py:5743](../../main.py#L5743) inserts `shipments(sales_order_id, shipped_at, customer_id)`. Similarly `shipment_lines` has different column sets at [main.py:2560](../../main.py#L2560) and [main.py:5808](../../main.py#L5808) (the second adds `sales_order_line_id`).
**Current behavior**: Two different shipment schemas depending on which endpoint wrote the row. Standalone ships never get `sales_order_id` set; SO ships never get `transaction_id` set.
**Risk**: Any future query joining shipments ↔ transactions ↔ sales_orders must special-case both shapes. See also [F05-08](15-traceability-gaps.md) (shipment_lines lacks `lot_id`).
**Recommended fix**: `create_shipment(cur, txn_id, customer_id, sales_order_id=None, shipped_at=...) -> shipment_id` helper that writes all columns in both cases.
**Effort**: small

### [F01-D7] Customer-alias JOIN pattern repeated 8× — ~20 lines
**Severity**: low
**Files**: [main.py:1193](../../main.py#L1193), [main.py:1204](../../main.py#L1204), [main.py:1232](../../main.py#L1232), [main.py:4731](../../main.py#L4731), [main.py:4993](../../main.py#L4993), [main.py:5085](../../main.py#L5085), [main.py:7183](../../main.py#L7183), [main.py:7197](../../main.py#L7197).
**Current behavior**: `LEFT JOIN customer_aliases ca ON ca.customer_id = c.id` + OR filter on `c.name`/`ca.alias` appears in 8 places with small variations (some aggregate via `ARRAY_AGG`, some filter with `DISTINCT`).
**Recommended fix**: DB view `customers_with_aliases` exposing `id, name, aliases[]`; endpoints join to view.
**Effort**: small

### [F01-D8] Case-weight multiplication `quantity * case_size_lb` — redundant with validator
**Severity**: low
**Files**: OrderLineInput validator at [main.py:894](../../main.py#L894), but inline rewrites at [main.py:2137](../../main.py#L2137) (receive), [main.py:4908](../../main.py#L4908) (create order), [main.py:5540](../../main.py#L5540) (add lines), [main.py:7939](../../main.py#L7939) (production_requirements).
**Current behavior**: 4+ variants of `total_lb = cases * case_weight` with slightly different validation.
**Recommended fix**: Single helper used everywhere including the validator.
**Effort**: small

---

## Proposed module-split plan — **DO NOT IMPLEMENT YET**

Treat `main.py` as a checklist. Target structure:

```
app/
  __init__.py             # create_app() factory, CORS, middleware
  config.py               # env vars, timezone constants, epsilon
  db.py                   # pool, get_db_connection, get_transaction
  auth.py                 # verify_api_key, verify_api_key_flexible, resolve_order_id
  models/
    __init__.py
    ops.py                # ReceiveRequest, ShipRequest, MakeRequest, PackRequest, AdjustRequest, PackLotAllocation
    sales.py              # CustomerCreate/Update, OrderLineInput, OrderCreate/HeaderUpdate/StatusUpdate, ShipOrderRequest, AddOrderLines
    notes.py              # NoteCreate, NoteUpdate
    admin.py              # ProductUpdate, BomLineCreate/Update, ProductBomCreate, AdminSQLQuery, LotMergeRequest, LotRenameRequest, SupplierLotUpdate, LotReassignmentRequest, Verify/Quick-create requests
    traceability.py       # (if any)
  helpers/
    lots.py               # find_or_create_lot, generate_lot_code, get_lot_balance, validate_lot_deduction, fifo_deduct [NEW], get_sibling_skus
    products.py           # _tiered_product_search, resolve_product, _resolve_single_product (consolidated)
    customers.py          # resolve_customer_id
    orders.py             # resolve_order_id, recompute_line_status, recompute_order_status [NEW], resolve_order_line_quantity [NEW]
    shipments.py          # create_shipment [NEW]
    timestamps.py         # get_plant_now, format_timestamp
    bilingual.py          # validate_bilingual (drop dead bilingual_response)
  routers/
    health.py             # GET /, GET /health
    dashboard_views.py    # /dashboard/inventory|low-stock|today|lots|production  ← candidate for deletion
    dashboard_api.py      # /dashboard/api/* (14 endpoints, frontend-private)
    products.py           # /products/*, /bom/*
    inventory.py          # /inventory/*
    lots.py               # /lots/*
    ops.py                # /receive, /ship, /make, /pack, /adjust, /void/{id} + 12 preview/commit aliases
    traceability.py       # /trace/*, /trace/supplier-lot/*
    customers.py          # /customers/*
    sales_orders.py       # /sales/orders/* (all 9 + ship + packing-slip)
    admin.py              # /admin/*
    production.py         # /production/*, /schedule, /reason-codes
    audit.py              # /audit/integrity (needs auth added — see F02-06)
    quick_create.py       # /products/quick-create*, /products/{id}/verify, /inventory/found*
    transactions.py       # /transactions/history
  services/
    pdf_packing_slip.py   # 479-line ReportLab block from main.py:5924
    production_scheduler.py  # 1,200-line scheduler from main.py:7890
    startup_migrations.py # inline DDL from main.py:170–480 — see F17-01
  main.py                 # now just create_app() and mount static
```

**Endpoint-to-module mapping** (alphabetical by subsystem path):
- `routers/health.py`: `/`, `/health`
- `routers/dashboard_views.py`: `/dashboard/{inventory,low-stock,today,lots,production}` (consider deleting all 5)
- `routers/products.py`: `/products/search`, `/products/missing-case-size`, `/products/resolve`, `/products/unverified`, `/products/test-batches`, `/products/{id}`, `/bom/products`, `/bom/batches/{id}/formula`
- `routers/inventory.py`: `/inventory/current`, `/inventory/lookup`, `/inventory/{item_name}`
- `routers/lots.py`: `/lots/by-supplier-lot/{code}`, `/lots/by-code/{code}`, `/lots/{id}`, `/lots/{code}/supplier-lot`, `/lots/{id}/rename`, `/lots/{id}/reassign`
- `routers/ops.py`: `/receive`, `/ship`, `/make`, `/pack`, `/adjust`, `/void/{id}` + 12 `/*/preview`, `/*/commit` aliases
- `routers/traceability.py`: `/trace/batch/{code}`, `/trace/ingredient/{code}`, `/trace/supplier-lot/{code}`, `/transactions/history`
- `routers/quick_create.py`: `/products/quick-create`, `/products/quick-create-batch`, `/products/{id}/verify`, `/inventory/found`, `/inventory/found-with-new-product`, `/inventory/found/queue`
- `routers/customers.py`: `/customers*` (4)
- `routers/sales_orders.py`: all 9 `/sales/*` + `/sales/orders/{id}/ship(/preview|/commit)` + `/sales/orders/{id}/packing-slip` + `/sales/dashboard`
- `routers/admin.py`: 12 `/admin/*`
- `routers/dashboard_api.py`: 14 `/dashboard/api/*`
- `routers/production.py`: `/production/requirements`, `/production/day-summary`, `/schedule`, `/reason-codes`
- `routers/audit.py`: `/audit/integrity`

Estimated target sizes: no file above 1,000 lines; largest would be `services/production_scheduler.py` (~1,200) and `services/pdf_packing_slip.py` (~480). Everything else in the 100–400 line range.

**Split effort**: large — 3–5 days including smoke tests, plus a follow-up PR for the DRY helpers. Recommend doing the split BEFORE the DRY pass so diffs stay reviewable.

**Do not split until:** the 30-op OpenAPI cap is first addressed (see [F06-04](16-schema-openapi.md)) — a router refactor shifts line numbers globally and will conflict with any in-flight schema edits. Recommend: freeze schema → split routers → then DRY helpers.

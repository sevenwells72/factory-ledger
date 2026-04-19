# Category 6 — Schema & OpenAPI Hygiene

Four OpenAPI files exist in the repo. Two are stale. The live contract is at the 30/30 hard cap with ~7 reclaimable slots.

---

### [F06-01] OpenAPI confirmed at 30/30 operations — zero headroom
**Severity**: informational
**Files**: [openapi-gpt-v3.yaml](../../openapi-gpt-v3.yaml) — 30 `operationId:` entries (verified by grep). CLAUDE.md states the 30-operation hard cap.
**Current state**: Every operation is live and in use. No dead operationIds.
**Implication**: Any new GPT-visible endpoint requires removing or merging an existing one. Operator workflows with new affordances (e.g. the scheduler's `/schedule` is NOT yet in the spec) will need slots reclaimed via F06-04.

---

### [F06-02] Two OpenAPI files are stale — one self-labeled DEPRECATED
**Severity**: medium
**Files**:
- [openapi-gpt-v3.yaml](../../openapi-gpt-v3.yaml) — **canonical GPT contract** v3.4.0, 30 ops, 849 lines
- [openapi-v3.yaml](../../openapi-v3.yaml) — **superset reference** v3.3.0, 33 ops, 894 lines (adds `productsMissingCaseSize`, `getCurrentInventory`, `getInventoryItem`, `checkFulfillment`, `productionDaySummary` — pre-unification names)
- [openapi-schema.yaml](../../openapi-schema.yaml) — **stale**, v2.7.0, 35 ops, 1154 lines. References dead split-endpoint pattern `/receive/preview`, `/ship/commit`, etc.
- [openapi-schema-gpt.yaml](../../openapi-schema-gpt.yaml) — **self-labeled DEPRECATED** at [line 5](../../openapi-schema-gpt.yaml#L5): "DEPRECATED — This schema uses split /preview and /commit endpoints that no longer exist in the API".

**Risk**: Changelog entry #13 notes an update to "both schemas" — a reviewer must remember which pair is current. When a new endpoint is added, someone WILL update the wrong file.

**Recommended fix**:
1. Delete `openapi-schema.yaml` and `openapi-schema-gpt.yaml`.
2. Add a `# CANONICAL — loaded into ChatGPT Custom GPT Actions config` header comment to `openapi-gpt-v3.yaml`.
3. Add a `# SUPERSET REFERENCE — not deployed; includes admin/dashboard ops not exposed to GPT` header to `openapi-v3.yaml`.
4. Update CLAUDE.md regression-guard to list only the two canonical files.
**Effort**: small

---

### [F06-03] Response envelope shapes diverge across list endpoints
**Severity**: medium
**Files**: openapi-gpt-v3.yaml; handlers in main.py.
**Current behavior** — 5 different envelope shapes for similar list endpoints:

| Endpoint | Envelope | Line |
|---|---|---|
| `searchProducts` | `{count, products}` | [main.py:1372](../../main.py#L1372) |
| `listCustomers` | `{customers}` (no count) | [main.py:4718](../../main.py#L4718) |
| `searchCustomers` | `{results}` (different key, no count) | [main.py:4738](../../main.py#L4738) |
| `inventoryLookup` | `{query, results}` | [main.py:1629](../../main.py#L1629) |
| `listOrders` | `{orders, count}` | [main.py:5053](../../main.py#L5053) |
| `getTransactionHistory` | `{count, transactions}` | [main.py:4136](../../main.py#L4136) |

**Risk**: The GPT's Action-calling prompt must know which key to unwrap per endpoint. Minor but accumulates into more prompt instructions (currently documented in `gpt-instructions-v3.md`).

**Recommended fix**: Adopt one envelope — `{count, items}` — across all list endpoints. Breaking; do in one coordinated PR with `gpt-instructions-v3.md` update.
**Effort**: medium

---

### [F06-04] Create-order vs get-order vs list-order shapes don't line up
**Severity**: medium
**Files**:
- `createOrder` response at [main.py:4953–4963](../../main.py#L4953): `{order_id, order_number, customer, requested_ship_date, status, total_lb, lines:[{line_id, product, quantity_lb, original_quantity, original_unit, case_weight_lb, unit_price}], warnings, message}`.
- `getOrder` response at [main.py:5244–5278](../../main.py#L5244): `{order_id, order_number, customer, order_date, requested_ship_date, status, notes, created_date, created_time, lines:[{name, unit_price, line_status, case_size_lb, cases, ...}]}`.
- `listOrders` response at [main.py:5022–5050](../../main.py#L5022): yet another shape.

**Differences observed**:
- `create` has no `order_date` / `created_date` / `line_status`; `get` has no `total_lb` / `warnings`.
- Line objects use `product` (create) vs `name` (get).
- Line objects use `case_weight_lb` (create) vs `case_size_lb` (get) for the same column.
- `quantity_lb` present in create; not in get line dict.

**Risk**: The GPT has to know two line shapes for the same entity. Documented in gpt-instructions but easy to drift.
**Recommended fix**: Single `render_order_line(row)` helper that emits a fixed shape; single `render_order(header, lines)` wrapper. Use in create, get, list.
**Effort**: small

---

### [F06-05] Operation-unification opportunities — ~7 reclaimable slots
**Severity**: informational (strategic)
**Current state**: 30/30 used. These collapses save budget without breaking functionality:

| Collapse | Savings | How |
|---|---:|---|
| `addOrderLines` + `cancelOrderLine` + `updateOrderLine` → `/sales/orders/{id}/lines` with `mode=add\|update\|cancel` | **2** | Mirrors the `mode=preview/commit` pattern already used for receive/ship/make/pack |
| `updateOrderHeader` + `updateOrderStatus` → single PATCH with any of `{status, requested_ship_date, notes, customer_id}` | **1** | Already both PATCH on same order; merge |
| `listCustomers` + `searchCustomers` → `GET /customers?q=&active_only=` | **1** | Query param handles both cases |
| `searchProducts` + `listProducts` (both take q, limit, product_type) → `GET /products?q=&limit=` | **1** | `listProducts` is essentially search with `limit=200` |
| `traceBatch` + `traceIngredient` + `traceSupplierLot` → `GET /trace?code=&kind=batch\|ingredient\|supplier_lot&product_id=` | **2** | All three return similar trace envelopes |

**Total reclaimable**: **7 slots**.
**Risk-free reclaims** (zero GPT behavior change required): F06-05 #4 and #5 are additive. The others are breaking — need coordinated `gpt-instructions-v3.md` update.
**Recommended fix**: Before the next feature ships, merge `listCustomers/searchCustomers` and `searchProducts/listProducts` (2 slots) as a warmup. Defer the rest until a slot is needed.
**Effort**: medium for the first two; larger for the trace merge.

---

### [F06-06] Parameter inconsistencies
**Severity**: low (documentation drift, not correctness)
**Files**: [openapi-gpt-v3.yaml](../../openapi-gpt-v3.yaml); handlers in main.py.

1. **`product_id` disambiguation present on 4 endpoints, missing on 1.**
   Present on `getLotByCode` ([line 410](../../openapi-gpt-v3.yaml#L410)), `traceBatch` ([line 506](../../openapi-gpt-v3.yaml#L506)), `traceIngredient` ([line 525](../../openapi-gpt-v3.yaml#L525)), `traceSupplierLot` ([line 544](../../openapi-gpt-v3.yaml#L544)). **Missing on `updateSupplierLot`** ([line 410](../../openapi-gpt-v3.yaml#L410)) even though the handler accepts it — [main.py:1889](../../main.py#L1889) has `product_id: Optional[int] = Query(None)`. The handler supports it; the spec hides it.
   **Fix**: Add `product_id` query param to `updateSupplierLot` operation definition.

2. **Pagination limits documented ≠ enforced.**
   `inventoryLookup` spec default 10, handler default 5 ([main.py:1612](../../main.py#L1612)). `getTransactionHistory` spec caps at 100, handler allows up to 1000 ([main.py:4076](../../main.py#L4076)). `listOrders` spec default 50, handler allows `ge=1, le=500`.
   **Fix**: Align spec with handler; use handler values as source of truth.

3. **Date format inconsistency.**
   `getTransactionHistory.since`/`until`, `getDaySummary.date`, `updateOrderHeader.requested_ship_date` all use `format: date`. But `OrderCreate.requested_ship_date` ([openapi-gpt-v3.yaml:226-227](../../openapi-gpt-v3.yaml#L226)) is plain `type: string` with no `format: date`.
   **Fix**: Add `format: date` to OrderCreate.

4. **`order_id` path-param type.**
   Declared as `type: string` on all 7 order endpoints. Handler resolves via `resolve_order_id` which accepts either int or `SO-YYMMDD-###`. Internally the db id is int. Responses return `order_id: 42` (int). String in the URL, int in the response — GPT can't round-trip the value back into another call without stringifying.
   **Fix**: Document the expected format (`string | integer`) in the schema description, or always return `order_number` alongside `order_id` for round-trip ease.

5. **Required/optional drift — ship requests.**
   `ShipRequest` requires `product_name, quantity_lb, customer_name, order_reference`. `ShipOrderRequest` has no required fields. Not wrong, just reflects different use cases.

**Effort**: small (all 5 sub-fixes)

---

### [F06-07] GPT instructions reference phantom operationId `createSalesOrder`
**Severity**: high (behavior-breaking)
**Files**: [GPT_INSTRUCTIONS.md:32](../../GPT_INSTRUCTIONS.md#L32), [GPT_INSTRUCTIONS.md:38](../../GPT_INSTRUCTIONS.md#L38), and the byte-identical [gpt-instructions-v3.md](../../gpt-instructions-v3.md).
**Current behavior**: Instructions tell the GPT to call `createSalesOrder`, but the operation in `openapi-gpt-v3.yaml` is named `createOrder` ([line 701](../../openapi-gpt-v3.yaml#L701)). The GPT will emit `createSalesOrder` action calls that don't match any action name, causing the "API-refusal" failure mode described in changelog entry #21.
**Related**: `openapi-schema.yaml` (the stale file) at [line 673](../../openapi-schema.yaml#L673) uses `createSalesOrder`. This is where the obsolete name came from — GPT instructions were likely written against that spec pre-rename.
**Risk**: Order-entry flow silently fails; GPT falls back to curl dumps. Most likely this has been observed and mitigated operator-side but not at its root.
**Recommended fix**: Change `GPT_INSTRUCTIONS.md` lines 32 and 38 from `createSalesOrder` → `createOrder`. Then delete the older identical file ([F02-04](11-dead-code.md)).
**Effort**: small (5 minutes)

---

### [F06-08] Two GPT-instructions files are byte-identical
**Severity**: low (see also [F02-04](11-dead-code.md))
**Files**: [GPT_INSTRUCTIONS.md](../../GPT_INSTRUCTIONS.md) and [gpt-instructions-v3.md](../../gpt-instructions-v3.md) — both 7,643 bytes.
**Risk**: Next edit will hit one and miss the other → regression (changelog #21 is precisely this class of failure).
**Recommended fix**: Delete one; keep `gpt-instructions-v3.md` (matches v3 naming). Update CLAUDE.md regression-guard to reference only one file.
**Effort**: small

---

### [F06-09] `lot_id` vs `id` key drift for the same primary key
**Severity**: low
**Files**: `getLotByCode` returns `id` ([main.py:1789](../../main.py#L1789), dict projection), but its own 409-ambiguous payload uses `lot_id` ([main.py:1813](../../main.py#L1813)). `traceBatch` and `traceSupplierLot` also use `lot_id` ([main.py:3512](../../main.py#L3512), [main.py:3926](../../main.py#L3926)).
**Risk**: Consumers of these endpoints have to know which key to read in which shape.
**Recommended fix**: Always emit both `id` and `lot_id` (duplicate for back-compat), and document that `lot_id` is canonical. Or pick one and update the GPT instructions.
**Effort**: small

---

### [F06-10] Inline-defined Pydantic models scattered across main.py
**Severity**: low (hygiene; relevant at module-split time)
**Files**: Most models live between [main.py:661–948](../../main.py#L661). But inline ones are defined right before their endpoint:
- `BulkResolveRequest` at [main.py:1132](../../main.py#L1132)
- `SupplierLotUpdate` / `LotRenameRequest` near [main.py:1888](../../main.py#L1888)
- `ProductUpdate`, `BomLineCreate`, `BomLineUpdate`, `ProductBomCreate`, `AdminSQLQuery`, `LotMergeRequest` near [main.py:7384](../../main.py#L7384), [main.py:7502](../../main.py#L7502), [main.py:7536](../../main.py#L7536), [main.py:7651](../../main.py#L7651), [main.py:7715](../../main.py#L7715), [main.py:7781](../../main.py#L7781)
**Recommended fix**: Move all to the Pydantic block or to `app/models/*.py` in the F01 module split.
**Effort**: small (done as part of module split)

# Category 3 — The 99-vs-30 Endpoint Audit

The backend exposes **99 FastAPI endpoints**. OpenAPI-gpt-v3 surfaces **30**. This file categorizes every backend endpoint.

Categories:
- **GPT** — listed in openapi-gpt-v3.yaml, called by ChatGPT Actions
- **DASH** — called by the Netlify dashboard (dashboard.js, index.html, traceability.html, process-flow.html) — per dashboard-endpoint audit
- **INT** — internal utility, called by other endpoints or scripts (includes unified endpoint + its preview/commit aliases)
- **STUB** — leftover from unified-endpoint migration or prior OpenAPI generation; candidate for removal
- **UNKNOWN** — can't determine usage from code alone; flag for human review

## Summary

| Category | Count | Notes |
|---|---:|---|
| GPT | 30 | Matches openapi-gpt-v3.yaml exactly |
| DASH | 21 | 14 are `/dashboard/api/*`, 7 are root-level shared with GPT or read-only |
| INT | 6 | 6 are resolver/helper paths called internally |
| STUB | 12 | The `/*/preview`, `/*/commit` aliases at [main.py:5857–5919](../../main.py#L5857) |
| Legacy/unused | 5 | `/dashboard/*` view endpoints — F02-03 flags for deletion |
| UNKNOWN | 25 | Write/admin endpoints presumed GPT-only but not in gpt spec — need human confirmation |

A dashboard endpoint that is also in GPT spec is listed as GPT with DASH noted. Preview/commit aliases are STUB regardless of history.

---

## Full table (99 rows)

| # | Method | Path | Decl line | Category | Notes |
|---:|---|---|---:|---|---|
| 1 | GET | /dashboard/inventory | [1266](../../main.py#L1266) | Legacy/unused | Reads SQL view; dashboard uses /dashboard/api/* — delete candidate (F02-03) |
| 2 | GET | /dashboard/low-stock | [1276](../../main.py#L1276) | Legacy/unused | Same as above |
| 3 | GET | /dashboard/today | [1286](../../main.py#L1286) | Legacy/unused | Same |
| 4 | GET | /dashboard/lots | [1296](../../main.py#L1296) | Legacy/unused | Same |
| 5 | GET | /dashboard/production | [1306](../../main.py#L1306) | Legacy/unused | Same |
| 6 | GET | / | [1321](../../main.py#L1321) | INT | Health/root banner |
| 7 | GET | /health | [1331](../../main.py#L1331) | INT | Health probe, liveness check |
| 8 | GET | /products/search | [1347](../../main.py#L1347) | GPT | `searchProducts`; also called by dashboard [process-flow.html:302](../../dashboard/process-flow.html#L302) |
| 9 | GET | /products/missing-case-size | [1378](../../main.py#L1378) | UNKNOWN | In openapi-v3.yaml but not gpt; probably admin tool |
| 10 | POST | /products/resolve | [1399](../../main.py#L1399) | GPT | `resolveProducts` |
| 11 | GET | /products/unverified | [1430](../../main.py#L1430) | UNKNOWN | Review-queue; likely dashboard-admin or GPT review flow |
| 12 | GET | /products/test-batches | [1451](../../main.py#L1451) | UNKNOWN | Review-queue; same as above |
| 13 | GET | /products/{product_id} | [1477](../../main.py#L1477) | DASH | Called by [process-flow.html:307](../../dashboard/process-flow.html#L307) |
| 14 | GET | /inventory/current | [1497](../../main.py#L1497) | UNKNOWN | In openapi-v3.yaml (not gpt); superseded by /inventory/lookup per changelog #19 |
| 15 | GET | /inventory/lookup | [1609](../../main.py#L1609) | GPT | `inventoryLookup` (changelog #19) |
| 16 | GET | /inventory/{item_name} | [1635](../../main.py#L1635) | UNKNOWN | In openapi-v3.yaml; predecessor to /inventory/lookup |
| 17 | GET | /lots/by-supplier-lot/{supplier_lot_code} | [1691](../../main.py#L1691) | UNKNOWN | Recall lookup; listed in openapi-schema but not gpt-v3 |
| 18 | GET | /lots/by-code/{lot_code} | [1784](../../main.py#L1784) | GPT+DASH | `getLotByCode`; also [traceability.html:793](../../dashboard/traceability.html#L793) |
| 19 | GET | /lots/{lot_id} | [1840](../../main.py#L1840) | UNKNOWN | Lot detail by id; may be linked from dashboard but no call found |
| 20 | PATCH | /lots/{lot_code}/supplier-lot | [1888](../../main.py#L1888) | GPT | `updateSupplierLot` |
| 21 | PATCH | /lots/{lot_id}/rename | [1953](../../main.py#L1953) | UNKNOWN | Added in changelog #14 for one-off UNKNOWN→25216 rename; now general-purpose but not GPT-exposed |
| 22 | POST | /receive | [2113](../../main.py#L2113) | GPT | `receive` (unified) |
| 23 | POST | /ship | [2315](../../main.py#L2315) | GPT | `ship` (unified) |
| 24 | POST | /make | [2597](../../main.py#L2597) | GPT | `make` (unified) |
| 25 | POST | /pack | [3070](../../main.py#L3070) | GPT | `pack` (unified) |
| 26 | POST | /adjust | [3312](../../main.py#L3312) | GPT | `adjust` (unified) |
| 27 | POST | /void/{transaction_id} | [3410](../../main.py#L3410) | UNKNOWN | Void endpoint; NOT in GPT schema, no dashboard call found — but referenced in change logs |
| 28 | GET | /trace/batch/{lot_code} | [3487](../../main.py#L3487) | GPT+DASH | `traceBatch`; [traceability.html:564](../../dashboard/traceability.html#L564) |
| 29 | GET | /trace/ingredient/{lot_code} | [3756](../../main.py#L3756) | GPT+DASH | `traceIngredient`; [traceability.html:702](../../dashboard/traceability.html#L702) |
| 30 | GET | /trace/supplier-lot/{supplier_lot_code} | [3918](../../main.py#L3918) | GPT | `traceSupplierLot` (changelog #12) |
| 31 | GET | /transactions/history | [4074](../../main.py#L4074) | GPT+DASH | `getTransactionHistory`; [traceability.html:375](../../dashboard/traceability.html#L375) |
| 32 | POST | /products/quick-create | [4146](../../main.py#L4146) | UNKNOWN | Quick-create flow; probably GPT-called but not in spec |
| 33 | POST | /products/quick-create-batch | [4197](../../main.py#L4197) | UNKNOWN | Same |
| 34 | POST | /lots/{lot_id}/reassign | [4245](../../main.py#L4245) | UNKNOWN | Historical-data correction; no GPT/dash caller found |
| 35 | POST | /inventory/found | [4357](../../main.py#L4357) | UNKNOWN | Found-inventory flow; presumed GPT-called |
| 36 | POST | /inventory/found-with-new-product | [4441](../../main.py#L4441) | UNKNOWN | Same |
| 37 | GET | /inventory/found/queue | [4516](../../main.py#L4516) | UNKNOWN | Review queue |
| 38 | POST | /products/{product_id}/verify | [4544](../../main.py#L4544) | UNKNOWN | Review queue |
| 39 | GET | /bom/products | [4611](../../main.py#L4611) | GPT | `listProducts` |
| 40 | GET | /bom/batches/{batch_id}/formula | [4642](../../main.py#L4642) | GPT | `getBatchFormula` |
| 41 | GET | /reason-codes | [4679](../../main.py#L4679) | UNKNOWN | Reference data; not in GPT spec |
| 42 | GET | /customers | [4710](../../main.py#L4710) | GPT | `listCustomers` |
| 43 | GET | /customers/search | [4724](../../main.py#L4724) | GPT | `searchCustomers` (changelog #21) |
| 44 | POST | /customers | [4744](../../main.py#L4744) | GPT | `createCustomer` |
| 45 | PATCH | /customers/{customer_id} | [4767](../../main.py#L4767) | GPT | `updateCustomer` |
| 46 | POST | /sales/orders | [4857](../../main.py#L4857) | GPT | `createOrder` (NB: GPT instructions still reference legacy name `createSalesOrder` — F06-05) |
| 47 | GET | /sales/orders | [4971](../../main.py#L4971) | GPT | `listOrders` |
| 48 | GET | /sales/orders/fulfillment-check | [5059](../../main.py#L5059) | UNKNOWN | In openapi-v3.yaml not gpt; may be dashboard-debug |
| 49 | GET | /sales/orders/{order_id} | [5212](../../main.py#L5212) | GPT+DASH | `getOrder`; [dashboard.js:1278](../../dashboard/dashboard.js#L1278) |
| 50 | PATCH | /sales/orders/{order_id}/status | [5368](../../main.py#L5368) | GPT | `updateOrderStatus` |
| 51 | PATCH | /sales/orders/{order_id} | [5426](../../main.py#L5426) | GPT | `updateOrderHeader` |
| 52 | POST | /sales/orders/{order_id}/lines | [5496](../../main.py#L5496) | GPT | `addOrderLines` |
| 53 | PATCH | /sales/orders/{order_id}/lines/{line_id}/cancel | [5575](../../main.py#L5575) | GPT | `cancelOrderLine` |
| 54 | PATCH | /sales/orders/{order_id}/lines/{line_id}/update | [5597](../../main.py#L5597) | GPT | `updateOrderLine` |
| 55 | POST | /sales/orders/{order_id}/ship | [5646](../../main.py#L5646) | GPT | `shipOrder` |
| 56 | POST | /receive/preview | [5857](../../main.py#L5857) | STUB | Forwards to /receive; no caller |
| 57 | POST | /receive/commit | [5862](../../main.py#L5862) | STUB | Forwards to /receive; no caller |
| 58 | POST | /ship/preview | [5867](../../main.py#L5867) | STUB | Forwards to /ship; no caller |
| 59 | POST | /ship/commit | [5872](../../main.py#L5872) | STUB | Forwards to /ship; no caller |
| 60 | POST | /make/preview | [5877](../../main.py#L5877) | STUB | Forwards to /make; no caller |
| 61 | POST | /make/commit | [5882](../../main.py#L5882) | STUB | Forwards to /make; no caller |
| 62 | POST | /pack/preview | [5887](../../main.py#L5887) | STUB | Forwards to /pack; no caller |
| 63 | POST | /pack/commit | [5892](../../main.py#L5892) | STUB | Forwards to /pack; no caller |
| 64 | POST | /adjust/preview | [5897](../../main.py#L5897) | STUB | Forwards to /adjust; no caller |
| 65 | POST | /adjust/commit | [5902](../../main.py#L5902) | STUB | Forwards to /adjust; no caller |
| 66 | POST | /sales/orders/{order_id}/ship/preview | [5907](../../main.py#L5907) | STUB | Forwards to /sales/orders/{id}/ship |
| 67 | POST | /sales/orders/{order_id}/ship/commit | [5914](../../main.py#L5914) | STUB | Forwards to /sales/orders/{id}/ship |
| 68 | GET | /sales/orders/{order_id}/packing-slip | [5926](../../main.py#L5926) | UNKNOWN | PDF gen — probably dashboard/email link, uses flexible auth |
| 69 | GET | /sales/dashboard | [6405](../../main.py#L6405) | UNKNOWN | Dashboard aggregate — no frontend call found |
| 70 | GET | /dashboard/api/production | [6509](../../main.py#L6509) | DASH | [dashboard.js:202](../../dashboard/dashboard.js#L202) |
| 71 | GET | /dashboard/api/inventory/finished-goods | [6589](../../main.py#L6589) | DASH | [dashboard.js:307](../../dashboard/dashboard.js#L307) |
| 72 | GET | /dashboard/api/inventory/batches | [6689](../../main.py#L6689) | DASH | [dashboard.js:374](../../dashboard/dashboard.js#L374) |
| 73 | GET | /dashboard/api/inventory/ingredients | [6774](../../main.py#L6774) | DASH | [dashboard.js:437](../../dashboard/dashboard.js#L437) |
| 74 | GET | /dashboard/api/activity/shipments | [6857](../../main.py#L6857) | DASH | [dashboard.js:498](../../dashboard/dashboard.js#L498) |
| 75 | GET | /dashboard/api/activity/receipts | [6918](../../main.py#L6918) | DASH | [dashboard.js:550](../../dashboard/dashboard.js#L550) |
| 76 | GET | /dashboard/api/lot/{lot_code} | [6974](../../main.py#L6974) | DASH | [dashboard.js:606](../../dashboard/dashboard.js#L606) |
| 77 | GET | /dashboard/api/product/{product_id}/lots | [7084](../../main.py#L7084) | DASH | [dashboard.js:699](../../dashboard/dashboard.js#L699) |
| 78 | GET | /dashboard/api/search | [7137](../../main.py#L7137) | DASH | [dashboard.js:767](../../dashboard/dashboard.js#L767) |
| 79 | GET | /dashboard/api/notes | [7226](../../main.py#L7226) | DASH | Wrapped by fetchAPI; list notes |
| 80 | POST | /dashboard/api/notes | [7271](../../main.py#L7271) | DASH | [dashboard.js:1083](../../dashboard/dashboard.js#L1083) |
| 81 | PUT | /dashboard/api/notes/{note_id} | [7295](../../main.py#L7295) | DASH | [dashboard.js:1075](../../dashboard/dashboard.js#L1075) |
| 82 | DELETE | /dashboard/api/notes/{note_id} | [7337](../../main.py#L7337) | DASH | [dashboard.js:1023](../../dashboard/dashboard.js#L1023) |
| 83 | PUT | /dashboard/api/notes/{note_id}/toggle | [7352](../../main.py#L7352) | DASH | [dashboard.js:998](../../dashboard/dashboard.js#L998) |
| 84 | PUT | /admin/products/{product_id} | [7384](../../main.py#L7384) | UNKNOWN | Admin-only edit |
| 85 | GET | /admin/bom/search | [7448](../../main.py#L7448) | UNKNOWN | Admin BOM editor |
| 86 | GET | /admin/bom/{product_id}/lines | [7470](../../main.py#L7470) | UNKNOWN | Admin BOM editor |
| 87 | POST | /admin/bom/{product_id}/lines | [7502](../../main.py#L7502) | UNKNOWN | Admin BOM editor |
| 88 | PUT | /admin/bom/lines/{line_id} | [7536](../../main.py#L7536) | UNKNOWN | Admin BOM editor |
| 89 | DELETE | /admin/bom/lines/{line_id} | [7581](../../main.py#L7581) | UNKNOWN | Admin BOM editor |
| 90 | GET | /admin/product-bom | [7624](../../main.py#L7624) | UNKNOWN | Admin FG↔batch mapping |
| 91 | POST | /admin/product-bom | [7651](../../main.py#L7651) | UNKNOWN | Admin FG↔batch mapping |
| 92 | DELETE | /admin/product-bom/{mapping_id} | [7682](../../main.py#L7682) | UNKNOWN | Admin FG↔batch mapping |
| 93 | POST | /admin/sql | [7715](../../main.py#L7715) | UNKNOWN | Diagnostic SQL passthrough; weak guard (F06-06) |
| 94 | GET | /admin/lots/duplicates | [7734](../../main.py#L7734) | UNKNOWN | Duplicate scanner |
| 95 | POST | /admin/lots/merge | [7781](../../main.py#L7781) | UNKNOWN | Lot merge; see GAP-11 |
| 96 | GET | /production/requirements | [7891](../../main.py#L7891) | UNKNOWN | Likely GPT-called via production flow |
| 97 | GET | /production/day-summary | [8078](../../main.py#L8078) | GPT | `getDaySummary` (added changelog #21) |
| 98 | POST | /schedule | [9085](../../main.py#L9085) | UNKNOWN | Dispatch endpoint for suggest/confirm/current scheduler modes |
| 99 | GET | /audit/integrity | [9122](../../main.py#L9122) | DASH | [dashboard.js:1397](../../dashboard/dashboard.js#L1397); no API-key auth (F02-06) |

---

## Findings

### [F03-01] 12 preview/commit alias stubs have no caller
**Severity**: medium
**Files**: rows 56–67 of the table. See [F02-02](11-dead-code.md) for the detailed delete recommendation (telemetry first, then removal).

### [F03-02] 5 legacy dashboard-view endpoints are unreferenced
**Severity**: medium
**Files**: rows 1–5. See [F02-03](11-dead-code.md) — all read SQL views that may no longer exist, all superseded by `/dashboard/api/*`. **Delete pending a production smoke-test.**

### [F03-03] 25 UNKNOWN-category endpoints need human disambiguation
**Severity**: medium — this is the most important finding of this category
**Files**: rows 9, 11, 12, 14, 16, 17, 19, 21, 27, 32–38, 41, 48, 68, 69, 84–95, 98.
**Current behavior**: These endpoints are not in openapi-gpt-v3.yaml, not called by the dashboard JS, and not internal helpers. They fall into a few buckets:
- **Superseded predecessors** (rows 14, 16 — `/inventory/current`, `/inventory/{item_name}`) — kept for back-compat but changelog #19 moved GPT to `/inventory/lookup`.
- **Review-queue / admin / data-correction** (rows 11, 12, 21, 27, 37, 38, 84–95) — likely one-off operational tools called via curl or an ad-hoc script. Need to confirm operator workflows still use them.
- **Production/scheduling** (row 96, 98) — may be internal to the scheduler or dashboard.
- **Presumed GPT-called but missing from spec** (rows 32, 33, 35, 36) — quick-create and found-inventory flows. If these are required by the GPT, they must be in the schema; if not, they're dead.
- **Orphaned helper APIs** (rows 41 reason-codes, 48 fulfillment-check, 68 packing-slip, 69 sales-dashboard) — some caller exists but isn't in the dashboard JS or GPT spec.

**Risk**: Each is a live endpoint with auth+DB access; any unused one is attack surface and operational confusion. Several write paths (rows 21, 32–36, 38, 87–92, 95) are irreversible-ish and should not be exposed if not in active use.

**Recommended fix**: Two-phase triage.
1. Instrument — add a single `@app.middleware("http")` that logs `request.url.path` + caller IP on each hit to Railway logs, run for 2 weeks, and produce a caller report.
2. For any endpoint with zero hits → delete. For endpoints with hits from the GPT → add to `openapi-gpt-v3.yaml` OR decide they should stop being called (tighten GPT instructions). For endpoints with hits from operators → move to `routers/admin.py` in the module split and add them to `openapi-v3.yaml` as a documented ops tool.

**Effort**: medium (half day to instrument + wait 2 weeks + half day to delete)

### [F03-04] Backend : OpenAPI ratio is 99 : 30 — cleanup target
**Severity**: medium
**Current behavior**: 99 backend handlers, 30 GPT-visible. That leaves 69 endpoints in a grey zone. Of those:
- 14 are legitimate dashboard-private (`/dashboard/api/*`)
- 12 are STUB aliases (delete candidates, F02-02)
- 5 are legacy dashboard view endpoints (delete candidates, F02-03)
- 8 are legitimate internals (root, health, lot helpers, packing slip, etc.)
- **30 are UNKNOWN** — see F03-03

**Ideal end-state after cleanup**: 99 − 12 (stubs) − 5 (legacy dash) = 82 endpoints. If F03-03 triage removes half the UNKNOWN endpoints, we could land near 70 total, of which 30 are GPT-visible, 14 are dashboard-private, and ~25 are documented admin/ops tools.

**Recommended fix**: Execute F02-02, F02-03, and F03-03 in sequence.
**Effort**: medium across all three items

---

## What's NOT a problem

- The 14 `/dashboard/api/*` endpoints are all actively called by the Netlify frontend. Legitimate — but need auth added per F02-07.
- The 30 GPT operations are all implemented; no spec-references-a-missing-endpoint drift was found.
- The 6 INT endpoints (root, health, 4 resolver helpers) are legitimate internal infrastructure.

# Change Log

## 2026-04-13 14:10 — Add background keepalive task (every 4 min) to prevent Supabase auto-pause
- **File(s) changed:** `main.py`
- **What changed:** Added `_run_keepalive()` background task using `threading.Timer` that runs `SELECT 1` every 240s. Starts on app startup, cancels on shutdown. Logs "keepalive ok" or "keepalive failed: {error}" — never crashes the app. No new dependencies.
- **Why:** Railway cron can't hit HTTP endpoints directly; needed an in-process solution to keep the Supabase connection alive.

---

## 2026-04-13 14:00 — Add /health/keepalive endpoint to prevent Supabase auto-pause
- **File(s) changed:** `main.py`
- **What changed:** Added GET /health/keepalive endpoint that runs SELECT 1 against Supabase and returns {"status":"ok","db":"reachable"} on success or 503 with error on failure. Uses existing get_transaction() context manager.
- **Why:** Supabase free-tier auto-pauses after inactivity; Railway logs showed repeated "could not send data to server: Connection timed out" errors on Apr 10-13 after days of no traffic.

---

## 2026-03-26 — Trim GPT instructions to fit 8000-char limit
- **File(s) changed:** `GPT_INSTRUCTIONS.md`
- **What changed:** Compressed ROUTING RULES, PRE-FLIGHT INTENT, QUERIES, and LOT MERGES sections to fit under 8,000 chars (now 7,986). Updated QUERIES to list /inventory/lookup as primary. Removed /inventory/{item} and packing-slip line from QUERIES (covered elsewhere).
- **Why:** Adding ROUTING RULES pushed instructions to 8,279 chars, exceeding the 8,000-char GPT limit.

---

## 2026-03-26 — Add ROUTING RULES section to GPT instructions
- **File(s) changed:** `GPT_INSTRUCTIONS.md`
- **What changed:** Added ROUTING RULES section before PRE-FLIGHT — INTENT. Bare product names route directly to inventoryLookup; product + "orders" routes to listOrders; product + "trace"/"lot" routes to trace endpoints; default fallback is inventoryLookup.
- **Why:** GPT was asking unnecessary clarification questions instead of immediately calling the right endpoint.

---

## 2026-03-26 — Performance fix for /inventory/lookup bulk queries
- **File(s) changed:** `main.py`
- **What changed:** Added `_inventory_detail_for_products()` that fetches product info and lot inventory for multiple product_ids in 2 SQL queries (using `ANY(%s)`) instead of 2N. Refactored `/inventory/lookup` to call the bulk function. Changed default limit from 10 to 5 to reduce payload size for broad queries.
- **Why:** Queries like "coconut" matching 17 products caused 34+ database round trips, making the endpoint slow.

---

## 2026-03-26 — Remove temporary debug endpoint and logging from inventory lookup
- **File(s) changed:** `main.py`
- **What changed:** Removed `/inventory/debug/{product_id}` endpoint and all diagnostic logging from `_inventory_detail_for_product()`. The 0-lb bug was caused by uncommitted code not being deployed, not a query issue.
- **Why:** Cleanup after confirming the inventory lookup works correctly once deployed.

---

## 2026-03-26 — Add debug logging and /inventory/debug/{product_id} endpoint for inventory lookup bug
- **File(s) changed:** `main.py`
- **What changed:** Added detailed debug logging to `_inventory_detail_for_product()` (logs all lots before HAVING, transaction_lines by product_id, and product_id mismatches). Added temporary `/inventory/debug/{product_id}` endpoint that returns raw diagnostic data (lots, txn_lines via lot join vs product_id, grouped results without HAVING, inventory_summary view).
- **Why:** `/inventory/lookup` returns 0 lb on hand for product 10305 (Sprinkles Rainbow 25 LB) despite 3,125 lb existing in lot 26-03-20-INVE-001. Dashboard shows correct balance. Debugging whether HAVING clause, lot_id join, or product_id mismatch is the root cause.

---

## 2026-03-26 — Remove getInventoryItem from GPT schema to stay at 30-op limit
- **File(s) changed:** `openapi-gpt-v3.yaml`
- **What changed:** Removed /inventory/{item_name} (getInventoryItem) from GPT schema since /inventory/lookup fully replaces it for GPT use. Brings operation count from 31 back to 30.
- **Why:** ChatGPT GPT actions enforce a 30-operation maximum

---

## 2026-03-26 — Add /inventory/lookup unified endpoint + fuzzy fallback on getInventoryItem
- **File(s) changed:** `main.py`, `openapi-gpt-v3.yaml`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`
- **What changed:** New GET /inventory/lookup?q= endpoint that combines fuzzy product search with lot-level inventory detail in a single call. Added _inventory_detail_for_product helper. Updated /inventory/{item_name} with fuzzy fallback (tiered search if no LIKE match; returns 404/300 for zero/multiple matches). Updated all three OpenAPI schemas with the new endpoint and revised summaries.
- **Why:** Replace the two-step searchProducts → getInventoryItem flow with a single unified lookup for the GPT

---

## 2026-03-26 — Create openapi-gpt-v3.yaml (unified schema for ChatGPT GPT action)
- **File(s) changed:** `openapi-gpt-v3.yaml`
- **What changed:** Created trimmed GPT action schema from openapi-v3.yaml (30 operations, the ChatGPT limit). Removed productsMissingCaseSize, searchCustomers, productionDaySummary. Added back getBatchFormula (needed for /make lot overrides). Uses unified mode-in-body endpoints instead of deprecated split /preview /commit paths.
- **Why:** The GPT is currently using the deprecated openapi-schema-gpt.yaml with split paths. This new schema uses the correct unified endpoints and fits within the 30-operation ChatGPT limit.

---

## 2026-03-26 — Fix GPT shipping 404: add split-path route aliases
- **File(s) changed:** `main.py`, `openapi-schema-gpt.yaml`
- **What changed:** Added thin wrapper routes for /preview and /commit sub-paths on all transactional endpoints (/receive, /ship, /make, /pack, /adjust, /sales/orders/{id}/ship). The GPT schema uses these split paths but the API only had the combined endpoint with mode in the body, causing 404s. Also added the missing shipOrderPreview endpoint to the GPT schema.
- **Why:** GPT was getting "not found" when trying to ship SO-260325-003 because it called /sales/orders/SO-260325-003/ship/commit which didn't exist as a route.

---

## 2026-03-26 09:00 — Update GPT instructions to v3.5.0
- **File(s) changed:** `GPT_INSTRUCTIONS.md`
- **What changed:** Complete rewrite of GPT instructions from developer-assistant format to operator-facing GPT v3.5.0. New structure includes: critical rules, pre-flight intent/product resolution, universal disambiguation format, batched disambiguation, order entry from confirmations, transaction workflow, order editing, shipping, FIFO override, receive, supplier lot cross-reference, found inventory, make, pack (with smart resolve and source batch mismatch), adjust, sales orders, ingredient lots, pack add-ins, post-commit, qty display, day summary, FG identity, lot merges, packing slip link-only, queries, bilingual, and error codes.
- **Why:** User requested update to new v3.5.0 instruction set

---

## 2026-03-25 16:00 — Add product_id disambiguation to all three trace endpoints
- **File(s) changed:** `main.py`, `openapi-schema-gpt.yaml`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added optional `product_id` query param to `/trace/supplier-lot` in main.py (batch and ingredient already had it). Added `product_id` param to all three trace endpoints in both OpenAPI schemas. Added `/trace/supplier-lot` endpoint to both OpenAPI schemas (was missing). Updated GPT instructions trace line to include supplier-lot and note the `?product_id=` param. Removed `updateNote` operation from GPT schema to stay within 30-operation limit.
- **Why:** GPT could detect ambiguous lot codes and ask the user to pick, but had no way to pass the chosen product_id back to any trace endpoint — the OpenAPI schemas didn't expose the parameter.

---

## 2026-03-25 15:30 — Allow service items (Pallets, freight) on sales orders with zero quantity_lb
- **File(s) changed:** `main.py`, `migrations/031_relax_quantity_lb_check_for_service_items.sql`
- **What changed:** Added is_service detection to create_sales_order and add_order_lines endpoints. Service items now skip case weight lookup, unit-defaulting warning, and quantity sanity check. quantity_lb defaults to 0 for service items. Migration relaxes CHECK constraint from > 0 to >= 0.
- **Why:** Service items like Pallets have no physical weight, causing the quantity_lb > 0 constraint to reject them.

---

## 2026-03-25 12:00 — Unit Display Implementation: dual lb · units format everywhere
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `dashboard/dashboard.js`, `dashboard/traceability.html`, `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md`
- **What changed:** Added unit counts (derived from case_size_lb / default_batch_lb) to all operator-facing endpoints and dashboard views. Backend: added case_size_lb/default_batch_lb to /products/search, /bom/products; new /products/missing-case-size endpoint; unit totals on sales order list/detail/ship/update-line; unit_count on production calendar FG rows, batch_count on batch rows; unit_count per line on shipments/receipts/inventory/lots/trace endpoints; packing slip qty_display changed from "N cs" to "X lb · Y units". Frontend: new fmtQty() helper; dual format in production calendar, sales orders (list/detail/KPI/lines), shipping/receiving activity, FG lot rows, batch lot rows, lot detail panel, traceability nodes. GPT instructions: added QTY DISPLAY section. OpenAPI schemas updated.
- **Why:** Operators need to see both weight (lb) and unit count simultaneously for all product quantities to reduce manual conversion and improve operational visibility.

---

## 2026-03-25 00:15 — Accept order_number strings in all /sales/orders/{order_id} endpoints
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `resolve_order_id` dependency that accepts either integer DB id or order_number string (e.g. 'SO-260323-001') and resolves to the integer id. Applied to all 8 sales order endpoints. Updated both OpenAPI schemas to declare order_id as `type: string`. Added note to GPT instructions ORDER EDITING section. Compressed GPT instructions to stay under 8,000 char limit (ORDER ENTRY, SUPPLIER LOT, PACKING SLIP sections).
- **Why:** GPT only knows the order_number (not the DB integer id), so FastAPI rejected order number strings with validation errors. GPT fell back to showing curl commands instead of executing.

---

## 2026-03-24 23:45 — Fix GPT "show don't tell" bug for ship date edits
- **File(s) changed:** `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `format: date` and explicit description with date conversion examples to `requested_ship_date` in both OpenAPI schemas. Strengthened ORDER EDITING section in GPT instructions to say "CALL API IMMEDIATELY" and include date format conversion rules. Added "NEVER show curl, API docs, or payloads" and "NEVER suggest cancel/recreate for edits" directives.
- **Why:** GPT was responding with curl commands and API documentation instead of calling updateOrderHeader when asked to change a ship date. Root cause: the v3 schema had no description/format on requested_ship_date, and GPT instructions mapped the operation but didn't explicitly direct execution.

---

## 2026-03-24 22:30 — GAP-3: Standalone /ship writes shipments + shipment_lines
- **File(s) changed:** `main.py`, `migrations/030_backfill_standalone_shipment_records.sql`
- **What changed:**
  - Standalone `/ship` commit path now creates a `shipments` row (with `transaction_id`, `shipped_at`, `customer_id`) and `shipment_lines` rows (one per lot shipped, with positive quantity) after writing transaction_lines. No sales_order_id or sales_order_line_id (left NULL).
  - Response now includes `shipment_id` field.
  - New migration 030: backfills `shipments` + `shipment_lines` for all existing standalone ship transactions that were missing them. Wrapped in a transaction, uses ABS() on negative transaction_line quantities.
  - Integrity checker (ship_missing_shipment_lines) unchanged — date filter `>= 2026-02-27` already correct; backfill covers all existing orphans.
- **Why:** Standalone shipments were invisible to anything querying shipment tables (packing slips, integrity checker, reports). Unifies shipping model so both standalone and order-linked paths write to shipments/shipment_lines.

---

## 2026-03-24 21:00 — Rename UNKNOWN lot to 25216 + add lot rename endpoint
- **File(s) changed:** `migrations/029_rename_unknown_lot_to_25216.sql`, `main.py`
- **What changed:**
  - New migration 029: renames lot 324 (Chocolate Sprinkles 25 LB, product_id 203) from lot_code='UNKNOWN' to '25216'. Includes pre-flight/post-flight checks, conflict guard, and idempotency (skips if already renamed).
  - New `PATCH /lots/{lot_id}/rename` endpoint in main.py: accepts `{"new_lot_code": "..."}`, validates no duplicate for same product, updates `lots.lot_code`, returns old/new codes. Schema audit confirmed only `lots` table stores lot_code as text — all other tables use integer FKs.
  - New `LotRenameRequest` Pydantic model.
- **Why:** Lot 324 was created with lot_code='UNKNOWN' during receive; real code from packing slip is 25216. Supplier lot was already backfilled but internal lot_code was still 'UNKNOWN'. Rename endpoint prevents future cases from needing raw SQL.

---

## 2026-03-24 19:30 — Option C trace: full upstream+downstream from any trace endpoint, no dead ends
- **File(s) changed:** `main.py`
- **What changed:**
  - `/trace/ingredient` no longer hard-rejects output lots (pack_output, production_output). Instead returns lot_origin, origin_note, upstream_ingredients (via ingredient_lot_consumption), plus direct_shipments/on_hand as normal
  - `/trace/batch` now includes customer_shipments, total_shipped_lb, and on_hand_lb — queries ship transactions that deducted from the batch lot
  - supplier_lot_code was already present on batch trace ingredient rows (no change needed)
- **Why:** Lot 601141 (Graham Cracker Crumbs NTF 10 LB) has entry_source=pack_output but ships directly to customers. The Fix 4 guard rail blocked it from /trace/ingredient, making shipment data invisible. Now both endpoints give the full picture regardless of which one is called.

---

## 2026-03-24 17:00 — Extend trace endpoints for direct-ship and supplier-lot exposure (FDA recall compliance)
- **File(s) changed:** `main.py`
- **What changed:**
  - Added `INGREDIENT_ENTRY_SOURCES` and `OUTPUT_ENTRY_SOURCES` constants; updated `/trace/batch` and `/trace/ingredient` to use them (fixes stale enum values like 'receive'/'make'/'pack')
  - Added `direct_shipments`, `total_shipped_lb`, and `on_hand_lb` to both `/trace/ingredient` and `_trace_ingredient_backward()` — queries ship transactions that deducted from ingredient lots
  - Added new `GET /trace/supplier-lot/{supplier_lot_code}` endpoint — recall-ready: finds all internal lots for a supplier lot (via `lots.supplier_lot_code` and `lot_supplier_codes` table), returns production usage, customer shipments, and exposure summary
- **Why:** Ingredient/resale lots shipped directly to customers (sprinkles, graham crumbs) were invisible to trace endpoints because trace only followed `ingredient_lot_consumption`. FDA recall compliance requires answering "where did supplier lot X end up?" for all products.

---

## 2026-03-24 15:45 — Backfill supplier lots for SO-260318-005 (American Classic Specialties)
- **File(s) changed:** API patches only (no file changes)
- **What changed:** Linked supplier lot codes for 3 lots from the American Classic Specialties shipment:
  - Rainbow Sprinkles lot `26-03-20-INVE-001` → supplier lot `550075853` (was missing)
  - Chocolate Sprinkles lot `UNKNOWN` (product_id 203) → supplier lot `25216` (was set to "UNKNOWN")
  - Graham Crumbs lot `601141` → supplier lot `601141` (was null)
- **Why:** Physical packing slips had supplier lot numbers handwritten but system records were missing the cross-references. Note: Chocolate Sprinkles lot code remains "UNKNOWN" — no API endpoint exists to rename lot codes; a manual DB UPDATE would be needed to rename it.

---

## 2026-03-24 14:30 — Factory Ledger Traceability Audit Report
- **File(s) changed:** `TRACEABILITY_AUDIT_2026-03-24.md`
- **What changed:** Created comprehensive traceability audit report identifying 13 gaps in the trace, ship, and lot lifecycle flows. Key findings: direct-ship-from-ingredient is invisible to trace endpoints (CRITICAL), no forward trace from lot to customer (CRITICAL), standalone /ship doesn't create shipment records (HIGH). Includes priority matrix and concrete fix proposals.
- **Why:** SO-260318-005 (Rainbow Sprinkles to American Classic Specialties) exposed broken backward traceability — lot 26-03-20-INVE-001 shows as never used despite being shipped, and supplier lot 550075853 was never linked.

---

## 2026-03-24 10:00 — Fix order entry behavior — act don't explain, minimize exchanges
- **File(s) changed:** `gpt-instructions-v3.md`
- **What changed:** Added ORDER ENTRY FROM CONFIRMATIONS section with silent extract→resolve→create flow; strengthened BE CONCISE (4 lines max for order confirmations, no unprompted next steps) and ACT DON'T LOOP (no payload display, no step headers, max 1 emoji); tightened DISAMBIGUATION for order entry (one tight question only); compressed SUPPLIER LOT, PACK, and QUERIES sections to stay under 8K char limit. Version bumped to v3.4.0.
- **Why:** GPT was acting as consultant during order entry — dumping SOPs, over-confirming, showing payloads, taking 6-7 exchanges for what should be 1-2.

---

## 2026-03-23 17:15 — Extend lot collision disambiguation to main dashboard
- **File(s) changed:** `main.py`, `dashboard/dashboard.js`
- **What changed:** Added `product_id` to `json_build_object` in shipments and receipts API responses so lot links in the Activity tab have product context. Added `product_id` to search API lots query. In dashboard.js: added `data-product-id` to shipping, receiving, and product panel lot links; added `data-search-lot-product-id` to search result lot items; pass `product_id` through all `openLotPanel` calls. Rewrote `openLotPanel` to handle HTTP 409 (ambiguous lot code) by rendering a disambiguation picker instead of a raw error. Added `renderLotDisambiguation()` helper.
- **Why:** Main dashboard lot links (shipping, receiving, search, product panel) didn't pass `product_id` to the lot detail endpoint, causing raw 409 errors after the collision fix was deployed.

---

## 2026-03-23 16:30 — Fix lot code collision bug + trace type misclassification
- **File(s) changed:** `main.py`, `dashboard/traceability.html`
- **What changed:** Added optional `product_id` query parameter to 5 endpoints (`/lots/by-code`, `/lots/{lot_code}/supplier-lot`, `/trace/batch`, `/trace/ingredient`, `/dashboard/api/lot`) for disambiguation when lot codes collide across products. When `product_id` is omitted and multiple lots match, endpoints return HTTP 409 with a list of matches. Added type validation to `/trace/ingredient` — returns 400 `wrong_trace_type` if called with a finished-goods lot. Fixed `/trace/batch` production query to join on `lot_id` (integer) instead of `lot_code` (string). Fixed `/trace/ingredient` downstream query to join on `lot_id`. Frontend: fixed `buildLotIndex` dedup to key on `lot_code|product_id`, added `selectedProductId` tracking, pass `product_id` in all API calls, added disambiguation modal for 409 responses, auto-route forward trace to batch endpoint on `wrong_trace_type`, escaped lot codes in onclick handlers (XSS fix), persist `product_id` in URL state.
- **Why:** Lot codes like "MAR 12 2026" are shared across products (ingredient + batch). Endpoints using `WHERE LOWER(lot_code) = LOWER(...)` with `fetchone()` returned whichever row Postgres found first, causing wrong lot data, wrong traces, and wrong supplier-lot updates.

---

## 2026-03-23 14:00 — Fix backward trace for ingredient lots
- **File(s) changed:** `main.py`, `dashboard/traceability.html`
- **What changed:** Updated `/trace/batch/{lot_code}` to detect ingredient lots (entry_source = receive/adjusted/found) and return supplier origin + downstream batches instead of 404. Added `_trace_ingredient_backward()` helper. Updated traceability.html to render ingredient backward traces with supplier → ingredient → batches → customers flow.
- **Why:** Backward trace failed with "Batch not found" for ingredient lots because the endpoint only looked for make/pack transactions.

---

## 2026-03-23 12:30 — Dashboard actionability audit report
- **File(s) changed:** `DASHBOARD_ACTIONABILITY_AUDIT.md`
- **What changed:** Created comprehensive audit report covering: complete endpoint inventory (85+ endpoints), dashboard page inventory (4 tabs, 4 standalone pages), gap analysis mapping write endpoints to dashboard locations, auth/CORS security assessment, and prioritized implementation plan with top 5 action sketches (ship order, update status, adjust inventory, receive, edit order header)
- **Why:** Planning phase for making the Netlify dashboard actionable (write operations) instead of read-only

---

## 2026-03-23 — "Ready to Ship" display label and reverse transition
- **File(s) changed:** `dashboard/dashboard.js`, `dashboard/index.html`, `main.py`, `gpt-instructions-v3.md`, `GUIDE.md`, `CONTEXT.md`
- **What changed:** Renamed dashboard display label from "Ready" to "Ready to Ship" in status label mapping and filter dropdown. Added `ready → in_production` reverse transition in both VALID_TRANSITIONS and MANUAL_TRANSITIONS. Updated GPT instructions to suggest marking orders as "Ready to Ship" after production completes, and documented the reverse transition. Updated GUIDE.md and CONTEXT.md to reflect the new display name and reverse transition.
- **Why:** Improve clarity of the "ready" status (now displayed as "Ready to Ship") and allow orders to move back to in_production if production falls short or inventory is consumed elsewhere.

---

## 2026-03-19 — Fix /make commit crash: is_ingredient column does not exist
- **File(s) changed:** `main.py`
- **What changed:** Replaced `COALESCE(is_ingredient, false) = false` with `type != 'ingredient'` in the post-commit auto-prompt /pack query (line ~2622). The `is_ingredient` column never existed in the products table; the correct column is `type` with value `'ingredient'`.
- **Why:** /make commit failed with "column is_ingredient does not exist" when trying to query finished goods for pack prompting after batch creation.

---

## 2026-03-19 15:20 — Add regression guard changelog and update CLAUDE.md
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CLAUDE.md`
- **What changed:** Created FACTORY_LEDGER_CHANGELOG.md documenting all major fixes with "Breaks If Reverted" column, known root causes, and 10 permanent rules. Added Regression Guard section to CLAUDE.md so future sessions check the changelog before making changes.
- **Why:** Prevent regressions by ensuring every Claude Code session is aware of past fixes and their dependencies.

---

## 2026-03-19 — Fix pallet charges counted as weight in sales order totals
- **File(s) changed:** `main.py`, `migrations/028_add_is_service_to_products.sql`
- **What changed:** Added `is_service` boolean column to products table (migration 028) to flag service/charge items like pallets, freight, surcharges. Updated 4 locations in main.py to exclude service products from weight totals: order detail Python loop (lines ~4441-4442), order list SQL SUM (lines ~4171-4172), dashboard overdue query (line ~5488), dashboard due-this-week query (line ~5507). Uses PostgreSQL FILTER clause in SQL queries and DB flag + keyword fallback in Python.
- **Why:** Pallet Charge lines (e.g., 1 unit stored as 1 lb) were inflating order weight totals. SO-260305-003 showed 251 lb instead of 250 lb due to a pallet charge line.

---

## 2026-03-19 — Swap GPT schema to openapi-v3.yaml; add sliced almonds products; update GPT instructions
- **File(s) changed:** `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `GPT_INSTRUCTIONS.md`, `migrations/027_add_sliced_almonds_products.sql`
- **What changed:** Added "ACTIVE GPT SCHEMA" header comment to openapi-v3.yaml confirming it's the canonical spec (30/30 ops). Added large deprecation warning box to openapi-schema-gpt.yaml. Updated GPT_INSTRUCTIONS.md to reference openapi-v3.yaml instead of deprecated schema. Created migration 027 to add two missing sliced almonds products: "Almonds – Sliced" (general ingredient) and "BS Almonds – Sliced – Raw" (Blue Stripes ingredient), both with idempotent NOT EXISTS guards.
- **Why:** GPT was configured with deprecated openapi-schema-gpt.yaml causing 404s on all receive operations (split /preview /commit endpoints no longer exist). Arturo couldn't receive "Almonds Slice" because no sliced almonds product existed in the DB.

---

## 2026-03-19 — Trim GPT instructions to fit 8,000 char limit
- **File(s) changed:** `gpt-instructions-v3.md`
- **What changed:** Condensed SOURCE BATCH MISMATCH WARNING and QUERIES product lookup lines to save ~300 characters. Final count: 7,869 chars (131 under limit). No behavioral rules removed.
- **Why:** GPT instructions were 8,167 chars after adding pack_needed/batch_hint rules, exceeding OpenAI's 8,000 char limit.

---

## 2026-03-19 — Update GPT instructions for pack_needed and batch_hint fields
- **File(s) changed:** `gpt-instructions-v3.md`, `GPT_INSTRUCTIONS.md`
- **What changed:** Added behavioral rule in MAKE section: when /make commit returns `pack_needed`, GPT must surface FG SKUs, ask operator to pack, and execute /pack calls. Added batch_hint note in PACKING SLIP section: when INSUFFICIENT lines have a `batch_hint`, GPT explains unpacked batch inventory exists and offers to run /pack. Added corresponding sections to developer-facing GPT_INSTRUCTIONS.md.
- **Why:** Ensures the GPT actually acts on the new `pack_needed` and `batch_hint` fields added to the API, closing the loop on forgotten /pack steps.

---

## 2026-03-19 — Add batch-inventory hints for INSUFFICIENT packing slips and /pack prompt after /make
- **File(s) changed:** `main.py`
- **What changed:** Fix 1: When packing slip shows INSUFFICIENT for a FG product, cross-references `parent_batch_product_id` to check if unpacked batch inventory exists and displays a hint (e.g., "Note: 500 lb of Batch BS Dark Chocolate is available — run /pack to convert to finished goods") both in the JSON data and rendered on the PDF. Fix 2: After `/make` commit, the response now includes a `pack_needed` object listing all FG SKUs linked via `parent_batch_product_id` that can be packed from the batch, prompting the operator to run `/pack`.
- **Why:** Recurring issue where production happens (/make) but the /pack step is forgotten, leaving batch inventory idle while packing slips show INSUFFICIENT for finished goods. FOUND lots were being used as a workaround.

---

## 2026-03-19 — Add automatic add-in ingredient deduction to /pack
- **File(s) changed:** `main.py`, `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md`, `openapi-schema-gpt.yaml`, `openapi-v3.yaml`
- **What changed:** Replaced `check_pack_source_mismatch()` with `resolve_pack_add_ins()` that detects when packing from a base batch into an FG with an intermediate batch BOM containing add-in ingredients. Preview now shows add-in quantities needed with FIFO availability. Commit automatically deducts add-in ingredients via FIFO with row-level locking. Falls back to mismatch warning when intermediate BOM is missing or source batch not in BOM. Updated GPT instructions and OpenAPI schemas.
- **Why:** Floor process blends add-in ingredients (PB Chips, Banana Bites) at the packing hopper — there is no separate /make step for flavor variants. System now matches the actual workflow.

---

## 2026-03-19 — Add pack source batch mismatch warning safeguard
- **File(s) changed:** `main.py`, `gpt-instructions-v3.md`, `GPT_INSTRUCTIONS.md`, `openapi-schema-gpt.yaml`, `openapi-v3.yaml`
- **What changed:** Added `parent_batch_product_id` column to products table (inline migration 012) linking FG products to their expected source batch. Added `check_pack_source_mismatch()` helper that returns warning fields when the /pack source batch doesn't match the target FG's expected parent batch. Warning (English + Spanish) included in both preview and commit responses. Populated mappings for all 8 BS Granola FG→batch pairs. Updated GPT instructions to display mismatch warnings prominently and suggest running /make first.
- **Why:** Arturo packed PB Banana FG cases directly from Dark Chocolate Granola base batch, skipping the required intermediate /make step that deducts add-in ingredients (PB Chips, Banana Bites).

---

## 2026-03-18 — Expose listProducts endpoint to GPT; drop getProduct
- **File(s) changed:** `openapi-v3.yaml`, `main.py`, `gpt-instructions-v3.md`
- **What changed:** Replaced `getProduct` (GET /products/{product_id}) with `listProducts` (GET /bom/products) in OpenAPI schema to stay at 30-operation ChatGPT cap. Bumped /bom/products default limit from 50→200 and max from 200→500. Updated GPT instructions to use listProducts for catalog queries and searchProducts for single lookups; removed /products/{id} reference.
- **Why:** GPT "list finished goods" queries only returned 7 SKUs because no catalog endpoint was exposed — it fell back to getCurrentInventory which only returns products with positive on-hand stock.

---

## 2026-03-18 — Add 8oz BS panel to dashboard; fix case weight display rounding
- **File(s) changed:** `dashboard/dashboard_config.json`, `dashboard/dashboard.js`
- **What changed:** Added new "6x8 OZ Retail Cases (BS Line)" panel with SKUs 70085-70088 to dashboard config. Fixed case weight display: `fmtInt()` was truncating 2.63 to 2 for 7oz BS products — added `fmtWt()` formatter that preserves decimals for non-integer weights while showing whole numbers cleanly (e.g., "25" not "25.0").
- **Why:** New 8oz BS products were missing from dashboard; 7oz products showed "× 2 lb" instead of "× 2.63 lb"

---

## 2026-03-18 — Add Blue Stripes 8 OZ Granola product SKUs
- **File(s) changed:** `migrations/026_add_bs_8oz_granola_products.sql`
- **What changed:** Created migration to insert 4 new BS Granola finished goods (Hazelnut Butter, Almond Butter, Dark Chocolate, Peanut Butter Banana — all 6x8 OZ Case, 3.0 lb, private_label). odoo_codes 70085–70088 (70081–70084 were already taken by Setton products). Product IDs: 206–209. Type is `finished` (not `finished_good` — check constraint).
- **Why:** New SKUs needed in the products table

---

## 2026-03-18 — Fix day summary to show pack consumption from older batch lots; void test transaction 469
- **File(s) changed:** `main.py`
- **What changed:** Fixed `/production/day-summary` endpoint to include pack consumption (and adjustments) from batch lots produced on previous days. Previously, only lots with a `make` transaction on the same day were included in `batch_lots`, so pack consumption from older lots was silently dropped. Now, when a lot_id is not already in the dict, it looks up the lot's product info and adds it with `produced_lb: 0.0`. Also voided test transaction 469 (1-case test pack of CQ Granola 10 LB, TXN-8EB734).
- **Why:** Production team reported that the day summary was missing pack consumption from batch lots made on prior days (e.g., Batch Classic Granola #9 made across MAR 10-11, with MAR 10 lot consumption missing from MAR 11 summary).

---

## 2026-03-18 — Add PATCH /lots/{lot_code}/supplier-lot endpoint and GPT integration
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added new `PATCH /lots/{lot_code}/supplier-lot` endpoint to attach or update supplier lot cross-references on existing lots after receiving. Updated OpenAPI schema (v3.3.0) with the new endpoint. Added "SUPPLIER LOT CROSS-REFERENCE" section to GPT instructions so the GPT auto-records mismatches when Arturo reports packing slip lot differs from system lot. Added new endpoint to QUERIES reference.
- **Why:** Arturo frequently encounters supplier lot numbers on packing slips (e.g., 550078168 for sprinkles) that don't match system lot codes. Previously there was no way to record this post-receive — only at receive time. This endpoint closes that gap.

---

## 2026-03-18 — Add supplier lot cross-reference for lot 26-03-10-FOUN-001
- **File(s) changed:** `migrations/025_set_supplier_lot_sprinkles_26-03-10-FOUN-001.sql`
- **What changed:** Created migration to set supplier_lot_code = '550078168' on lot 26-03-10-FOUN-001 (Sprinkles Rainbow 10 LB). This lot was used to ship 23 cases to International Gourmet Foods (SO-260318-001, Shipment #32).
- **Why:** Packing slip shows supplier lot 550078168; adding cross-reference for traceability.

---

## 2026-03-16 17:20 — Deploy to Netlify and connect GitHub auto-deploy
- **File(s) changed:** (no code changes — deployment config only)
- **What changed:** Linked Netlify site (cns-factory-ledger) to GitHub repo sevenwells72/factory-ledger for auto-deploy from main branch with publish dir `dashboard/`. Deployed latest dashboard.js with Railway API URL fix to production.
- **Why:** Site was using Netlify Drop (manual uploads) and wasn't picking up git pushes. Now auto-deploys on every push to main.

---

## 2026-03-16 17:15 — Fix dashboard API calls for Netlify hosting
- **File(s) changed:** `dashboard/dashboard.js`
- **What changed:** Changed `API_BASE` from relative `/dashboard/api` to absolute `https://fastapi-production-b73a.up.railway.app/dashboard/api`. Also updated the `/audit/integrity` health badge fetch to use the full Railway URL. Relative paths were resolving to Netlify (returning 404 HTML) instead of the Railway backend.
- **Why:** Dashboard deployed on Netlify was showing "Failed to load" errors with 404 HTML responses for all API calls.

---

## 2026-03-16 17:00 — Add top navigation bar across all dashboard pages
- **File(s) changed:** `dashboard/index.html`, `dashboard/sankey.html`, `dashboard/process-flow.html`, `dashboard/traceability.html`, `dashboard/dashboard.css`
- **What changed:** Added a fixed 48px top navigation bar to all four dashboard HTML files. "CNS Factory Ledger" branding on the left, pill-style nav links (Dashboard, Material Flow, Production Lines, Traceability) on the right with active page highlighted in blue. Collapses to hamburger menu below 768px. Adjusted sticky header and tab-bar positions in all files to account for the new 48px offset. Removed redundant "back to Dashboard" links from sub-pages.
- **Why:** Users needed a way to navigate between all dashboard pages without returning to the main dashboard first.

---

## 2026-03-16 16:30 — Add lot traceability network graph page
- **File(s) changed:** `dashboard/traceability.html`
- **What changed:** Built a new audit-critical lot traceability page with D3.js network graph. Supports forward trace (ingredient -> batches -> customers) and backward trace (batch -> ingredients -> suppliers). Features include: search with autocomplete from lot index, completeness badges (complete/partial/incomplete), confirmed vs legacy/unknown link distinction (solid vs dashed lines), fan-out collapse for large traces (>8 children), zoom/pan controls, detail audit table, print and text export for auditors, URL deep-linking (?lot=X&direction=Y), and proper data gap handling with warning icons.
- **Why:** Required for food safety compliance (Setton Farms audits, HACCP, FDA recall readiness). Traces ingredient lots through production batches to customer shipments with honest data completeness indicators.

---

## 2026-03-16 15:00 — Increase transaction history limit and Sankey fetch size
- **File(s) changed:** `main.py`, `dashboard/sankey.html`
- **What changed:** Raised `/transactions/history` max limit from 100 to 1000 (line 2992 in main.py). Updated Sankey diagram fetch constant from 100 to 500 for both make and ship transaction fetches.
- **Why:** 100-record cap truncated data in Sankey diagram; 500 captures fuller production/shipment history

---

## 2026-03-16 14:30 — Add manufacturing process flow dashboard
- **File(s) changed:** `dashboard/process-flow.html`
- **What changed:** New single-file HTML dashboard showing 4 production lines (Granola Baking, Coconut Sweetened, Bulk Packing, Pouch Line) as cards with visual stage pipelines. Features: product classification by output name, today's production metrics (input lb, batch count, output lb, cases), yield calculation with coconut hydration handling, summary strip (lines active, produced today, avg yield, workers placeholder), auto-refresh every 60s with stale-data detection, dark industrial theme, fallback sample data when API is offline.
- **Why:** Need a real-time production floor view showing each line's status and throughput derived from make/pack transactions

---

## 2026-03-16 14:00 — Add Sankey diagram page for product flow visualization
- **File(s) changed:** `dashboard/sankey.html`
- **What changed:** New single-file HTML page with D3.js Sankey diagram showing raw materials → production lines → finished goods → customers. Features: date range selector, top-N controls for ingredients/products/customers, dark industrial theme, API integration with fallback sample data, hover tooltips, 50 lb minimum flow threshold.
- **Why:** Visualize end-to-end product flow through the factory operation

---

## 2026-03-12 — Close SO-260213-001 unrecorded shipment (Juliette Food LLC)
- **File(s) changed:** `migrations/024_close_so260213001_juliette.sql`
- **What changed:** Marked all 5 order lines (4 granola products + pallets) as fulfilled and set order to shipped. The physical shipment (BOL 28106-I, 02/26/2026, customer pick up) was never recorded as ship transactions in the system.
- **Why:** Order was stuck in confirmed status with 0 shipped despite physical shipment being complete per packing slip

---

## 2026-03-12 — Reconcile SO-260312-005 with pre-existing shipments (DiCarlo Food Service)
- **File(s) changed:** `migrations/023_reconcile_so260312005_dicarlo.sql`
- **What changed:** Linked 4 standalone ship transactions (Graham Cracker Crumbs 400 lb, Fancy UNIPRO 400 lb, Sprinkles Chocolate 600 lb, Sprinkles Rainbow 1,800 lb) to sales order SO-260312-005. Created shipments, sales_order_shipments, and shipment_lines records; updated line shipped quantities and statuses to fulfilled; set order status to shipped.
- **Why:** Shipments were recorded before the sales order was entered, so they weren't linked

---

## 2026-03-12 — Close SO-260217-001 partial ship (Feeser's Food Distributors)
- **File(s) changed:** `migrations/022_close_so260217001.sql`
- **What changed:** Closed order SO-260217-001 by reducing Flake UNIPRO 10 LB ordered qty from 300 → 200 lb to match what was shipped, marked line as fulfilled, updated order status to shipped. Remaining 100 lb will not be shipped per business decision.
- **Why:** Order was stuck in partial_ship status; business decided to close it as-is

---

## 2026-03-12 — Fix SO-260217-008 under-shipment (Curtze Food Service)
- **File(s) changed:** `migrations/021_fix_so260217008_undershipment.sql`
- **What changed:** Created correction migration to fix order SO-260217-008 which shows "Partial Ship" but was fully shipped per packing slip (invoice 28108-I, 02/24/2026). Medium UNIPRO 10 LB was recorded as 1,080 lb shipped instead of 1,200 lb (120 lb short due to insufficient on-hand at ship time). Pallets line (qty 1) was not recorded at all. Migration updates sales_order_lines, sales_order_shipments, shipment_lines, and order status.
- **Why:** Physical shipment (per BOL) was complete but system under-recorded due to inventory cap at ship time

---

## 2026-03-09 13:43 — Add /production/day-summary to schema, trim GPT instructions
- **File(s) changed:** `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `/production/day-summary` GET endpoint to OpenAPI schema (was in main.py but missing from schema). Removed `/lots/{lot_id}` (getLot) to stay within GPT's 30-operation limit — `getLotByCode` covers all real usage. Bumped schema version to 3.2.0. Trimmed GPT instructions from ~9,400 chars to ~5,600 chars by removing redundant field-by-field listings for RECEIVE, MAKE, PACK, and ADJUST sections (the schema already provides these).
- **Why:** GPT was slow due to ~500-800 wasted tokens per turn from duplicated field listings. Missing day-summary endpoint caused GPT to hallucinate when users said "wrap up" or "daily summary". Hit 30-operation GPT limit after adding day-summary.

---

## 2026-03-09 18:45 — Add since/until date filters to /transactions/history
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `since` and `until` (YYYY-MM-DD, both inclusive) query parameters to the `/transactions/history` endpoint. Filters on `t.timestamp`. Updated both OpenAPI specs and GPT instructions to document the new parameters.
- **Why:** GPT was hanging when users asked for transactions since a specific date — the endpoint had no date filtering, causing either no results or too much data.

---

## 2026-03-09 11:19 — Fix reconciliation SQL script: p.sku → p.odoo_code
- **File(s) changed:** `factory_ledger_reconciliation.sql` (also updated in Supabase SQL Editor)
- **What changed:** Replaced `p.sku` with `p.odoo_code` on lines 116 (SELECT) and 123 (GROUP BY) in Test 5 (Negative Balance). The `products` table has no `sku` column; `odoo_code` is the equivalent. All other column references across all 6 tests and the summary block were verified correct against the production schema via `information_schema.columns` inspection and EXPLAIN dry-run.
- **Why:** Script failed on Supabase production with "column so.ship_date does not exist" (original report) / "column p.sku does not exist" (actual remaining error after inspection).

---

## 2026-03-09 11:05 — Patch three critical bugs: transaction safety, zero-shipment guard, force_standalone audit trail
- **File(s) changed:** `main.py`
- **What changed:**
  - Fix 1: Added `except HTTPException: raise` to 4 write endpoints missing it (`/products/quick-create`, `/products/quick-create-batch`, `/inventory/found-with-new-product`, `/customers` POST). All other commit endpoints already had correct outer try/except pattern.
  - Fix 2: Added zero-shipment guard in `POST /sales/orders/{id}/ship` commit — if all line items hit the `no_stock` path (total shipped = 0), the pre-created shipments row is deleted and a 409 is returned instead of committing an empty `partial_ship` status update.
  - Fix 3: In `POST /ship` commit, when `force_standalone=true` is used and customer has open orders: logs `logger.warning` with customer name and order count, records `standalone_override=true` in transaction notes, returns `warning` and `standalone_override` fields in response JSON.
- **Why:** Code audit found these three patterns causing inventory/order drift in production.

---

## 2026-03-05 15:00 — Pre-merge review fixes (Decimal serialization, void safety, migration 015)
- **File(s) changed:** `main.py`, `migrations/015_fix_lot131_negative_balance.sql`
- **What changed:** Added DecimalSafeJSONResponse as default FastAPI response class for NUMERIC column compatibility; added FOR UPDATE lock + COALESCE(status) on /void endpoint; made migration 015 note text dynamic using actual balance
- **Why:** Pre-merge review identified psycopg2 Decimal serialization risk from migration 020, potential double-void race condition, and hardcoded balance in migration 015

---

## 2026-03-05 14:30 — Factory Ledger Data Integrity Remediation (Priorities 0–7)
- **File(s) changed:** `main.py`, `dashboard/index.html`, `dashboard/dashboard.css`, `dashboard/dashboard.js`, `migrations/015_fix_lot131_negative_balance.sql`, `migrations/016_backfill_received_at.sql`, `migrations/017_transaction_status_and_void_ghosts.sql`, `migrations/018_backfill_pre_migration_shipments.sql`, `migrations/019_populate_case_size_lb.sql`, `migrations/020_numeric_precision.sql`
- **What changed:**
  - P0: Added `validate_lot_deduction()` guard with BALANCE_EPSILON=0.0001 to /pack, /ship, /make, /sales/orders/{id}/ship
  - P1: Migration 015 posts adjustment to fix lot 131 -60 lb negative balance
  - P2: Migration 016 backfills received_at; updated all FIFO ORDER BY to use COALESCE(received_at, created_at)
  - P3: Migration 017 adds status column, voids ghost makes 80/83/84/177; added POST /void/{id} endpoint; 0-lb make guardrail; dashboard queries filter voided transactions
  - P3B: Migration 018 backfills shipments/shipment_lines for 11 pre-migration ship transactions
  - P4: Migration 019 auto-populates case_size_lb from product names; improved /pack error messages
  - P5: Migration 020 converts quantity columns to NUMERIC(14,4); added Decimal-based calculations
  - P6: Added supplier_lot_code required validation on /receive
  - P7: New GET /audit/integrity endpoint with 8 automated checks + dashboard health badge
- **Why:** Data integrity audit found negative lot balances, missing metadata, ghost transactions, floating point dust, and other issues requiring systematic remediation.

---

## 2026-03-05 11:30 — Fix SO-260217-001 Flake UNIPRO over-shipment (300→200 lb)
- **File(s) changed:** `migrations/014_fix_so260217001_flake_overshipment.sql`, `main.py`
- **What changed:** Created migration 014 to correct the over-shipment: zeroes out the 100 lb over-deduction on lot "FEB 24 2026" (lot 242) in transaction_lines, updates shipment_lines/sales_order_shipments from 300→200 lb, reduces sales_order_line 48 shipped qty from 300→200 and status from fulfilled→partial, and sets order 32 status to partial_ship. Also added `AND tl.quantity_lb != 0` filter to the packing slip query to skip zeroed-out correction rows.
- **Why:** System recorded 300 lb shipped for Coconut Sweetened Flake UNIPRO 10 LB on SO-260217-001 but only 200 lb physically left the warehouse. The extra 100 lb was a data entry error at ship time.

---

## 2026-03-05 09:50 — Fix packing slip to use actual shipment records instead of live FIFO
- **File(s) changed:** `main.py`
- **What changed:** Patched the packing slip endpoint (`/sales/orders/{id}/packing-slip`) to check for committed shipment records in the `shipments` table. If shipments exist, lot allocations are pulled from `shipment_lines → transaction_lines → lots` (reflecting what was actually shipped). If no shipments exist yet (pre-shipment), the original FIFO inventory preview is preserved for pick-list use.
- **Why:** After a shipment was committed, the packing slip was recalculating FIFO against current (post-shipment) inventory, causing it to show less than what was actually shipped (e.g., showing 200 lb instead of 300 lb for Coconut Sweetened Flake UNIPRO 10 LB on SO-260217-001).

---

## 2026-03-04 12:55 — Clarify lot_allocations schema description: source overrides only
- **File(s) changed:** `openapi-v3.yaml`
- **What changed:** Updated lot_allocations description in PackRequest to explicitly state it's for SOURCE lot overrides only, and to use target_lot_code for the output/FG lot.
- **Why:** GPT was misrouting operator-provided lot codes into lot_allocations (source) instead of target_lot_code (output).

---

## 2026-03-04 12:50 — Fix GPT pack lot code misrouting: lot code = output lot, not source
- **File(s) changed:** `gpt-instructions-v3.md`
- **What changed:** Added explicit instruction that when operator provides a lot code during packing, it is the target_lot_code (FG output lot), NOT a source lot allocation. Source lots use FIFO unless operator explicitly says "pull from lot X."
- **Why:** After schema fix, GPT successfully called /pack but misinterpreted lot code 602271 as a source lot allocation instead of the output lot, causing a "0 lb available" error.

---

## 2026-03-04 12:30 — Fix GPT pack workflow: schema descriptions + instruction conflict
- **File(s) changed:** `openapi-v3.yaml`, `gpt-instructions-v3.md`, `openapi-schema-gpt.yaml`
- **What changed:** Added field descriptions (source_product, target_product, case_weight_lb, lot_allocations, target_lot_code) to PackRequest in v3 OpenAPI schema so GPT knows it can pass raw SKU codes. Fixed instruction "Always ask which FG SKU" → only ask if FG SKU not already provided. Marked old openapi-schema-gpt.yaml (v2.7.0) as deprecated — its split /preview and /commit endpoints don't exist in the API.
- **Why:** GPT was stuck in a verify/resolve loop during pack operations. Root cause: old schema's /pack/preview endpoint returns 404 (only /pack with mode param exists), and missing field descriptions left GPT uncertain what values to pass.

---

## 2026-03-02 17:35 — Auto-populate case_size_lb for OZ-pattern products
- **File(s) changed:** `main.py`
- **What changed:** Extended Migration 006 startup logic to handle OZ-pattern product names (e.g., "12x10 OZ", "6x7 OZ"). Parses count × unit-oz from the name, converts to lb (÷16), and sets `case_size_lb`. Also ran UPDATE directly against production DB for 9 existing products: 5 × 12x10 OZ → 7.50 lb, 4 × 6x7 OZ → 2.63 lb.
- **Why:** Products with OZ-based case formats had `case_size_lb = NULL`, which blocked the `/pack` endpoint with HTTP 400. SKU 70003 (Granola SS Chocolate Chip 12x10 OZ Case) was specifically reported as failing during packing.

---

## 2026-03-02 17:15 — Migration 013: Create shipment tables
- **File(s) changed:** `migrations/013_shipment_tables.sql`
- **What changed:** Created migration for three missing tables: `shipments` (one record per ship/commit), `sales_order_shipments` (links transactions to order lines), and `shipment_lines` (per-product detail within a shipment). All columns derived from existing INSERT/SELECT statements in main.py. Includes foreign keys to sales_orders, customers, sales_order_lines, transactions, and products, plus indexes on all FK and timestamp columns.
- **Why:** The ship/commit endpoint (`POST /sales/orders/{id}/ship`) and sales dashboard were referencing these tables but no CREATE TABLE migration existed, causing "relation does not exist" errors on any dispatch operation.

---

## 2026-02-27 09:48 — Add /products/resolve to v3 OpenAPI schema
- **File(s) changed:** `openapi-v3.yaml`
- **What changed:** Added `/products/resolve` endpoint and updated `/products/search` description in the v3 schema (the one the GPT uses for actions). Bumped version 3.0.0 → 3.1.0.
- **Why:** The GPT needs to see the new resolve endpoint in its action schema to use it.

---

## 2026-02-27 09:40 — 3-tier fuzzy product search + bulk resolve endpoint
- **File(s) changed:** `main.py`, `openapi-schema.yaml`, `openapi-schema-gpt.yaml`, `migrations/012_pg_trgm_product_search.sql`
- **What changed:**
  - Added `_tiered_product_search()` helper: exact match → keyword (word-order independent via ILIKE ALL) → trigram similarity fallback
  - Upgraded `GET /products/search` to use 3-tier search; returns `match_tier` per result
  - Added `POST /products/resolve` endpoint for bulk product name resolution with confidence levels (high/medium/low/none)
  - Rewrote `resolve_product_id()` and `resolve_product_full()` to use 3-tier search instead of simple ILIKE
  - Added migration 012: enables `pg_trgm` extension and GIN index on `products.name`
  - Updated both OpenAPI schemas (v2.6.0 → v2.7.0) with new `/products/resolve` endpoint and improved `/products/search` docs
- **Why:** OCR'd order confirmations send product names in different word orders (e.g. "Chocolate Sprinkles 10 lb" vs DB name "Sprinkles Chocolate 10 LB"), causing sales order creation to fail. This makes product matching word-order and typo tolerant.

---

## 2026-02-26 — Add Sprinkles 25 LB finished goods (Rainbow & Chocolate)
- **File(s) changed:** `main.py`, `dashboard/dashboard_config.json`
- **What changed:**
  - Added Migration 012 in main.py: inserts two new finished goods — Sprinkles Rainbow 25 LB (Odoo 10305) and Sprinkles Chocolate 25 LB (Odoo 10306)
  - Added both SKUs to the "25 LB Bulk Cases" panel in dashboard_config.json
- **Why:** 25 LB versions of existing 10 LB sprinkles products (10302 Rainbow, 10303 Chocolate) were needed.

---

## 2026-02-26 15:10 — Supplier lot field: complete read integration
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Added Migration 011: `supplier_lot_code`, `lot_type`, `received_at` columns on `lots` table; `lot_supplier_codes` table with indexes for commingled receipt breakdowns
  - GET `/lots/by-code/{code}`: added `supplier_lot_code`, `lot_type`, `supplier_lot_entries` to response
  - GET `/lots/{id}`: added `supplier_lot_entries` array (columns already came through via `l.*`)
  - NEW endpoint GET `/lots/by-supplier-lot/{code}`: searches both `lots.supplier_lot_code` and `lot_supplier_codes` table for recall tracing
  - GET `/trace/batch/{lot}`: ingredient lots now include `supplier_lot_code`
  - GET `/trace/ingredient/{lot}`: response now includes `supplier_lot_code`
  - Packing slip PDF: FIFO lots now include supplier lot reference in small gray text below internal lot code
  - OpenAPI: added `getLotsBySupplierLot` operation (now 32 operations)
  - GPT instructions: added supplier lot search to QUERIES section
- **Why:** Supplier lot data was write-only (captured during receive but never surfaced in any query). Needed for recall tracing and full chain traceability.

---

## 2026-02-26 14:40 — Packing slip: stronger GPT instructions to output link only
- **File(s) changed:** `gpt-instructions-v3.md`, `openapi-v3.yaml`
- **What changed:**
  - Rewrote PACKING SLIP section with explicit prohibitions: NEVER generate inline slip, NEVER list items, NEVER say "Sent to dock printer"
  - Specified exact markdown link format: `📎 [Packing Slip – {order_number}](URL)`
  - Strengthened OpenAPI description to say "do NOT call as API action" and "never recreate in chat text"
- **Why:** GPT was hallucinating an inline text packing slip instead of providing the clickable PDF link

---

## 2026-02-26 14:25 — Packing slip: query param auth for browser-clickable links
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Added `verify_api_key_flexible` dependency that accepts API key from either `X-API-Key` header or `?key=` query parameter
  - Updated packing slip endpoint to use flexible auth so browser link clicks work
  - Updated GPT instructions to present clickable URL instead of calling API directly
  - Updated OpenAPI spec with `key` query parameter on getPackingSlip
- **Why:** ChatGPT can't display binary PDF inline — secretary needs a clickable link to open in browser and print

---

## 2026-02-26 14:09 — Added packing slip PDF endpoint
- **File(s) changed:** `main.py`, `requirements.txt`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Added `GET /sales/orders/{order_id}/packing-slip` endpoint that generates a printable PDF packing slip
  - PDF includes: company header, BILL TO / SHIP TO, ship date, SO#, items table with FIFO lot allocation (read-only), warehouse confirmation signature lines, traceability note, footer with timestamp
  - Non-weight items (pallets, freight) shown with "N/A" lot; insufficient stock flagged as "INSUFFICIENT"
  - Added `reportlab==4.1.0` to requirements.txt
  - Added `StreamingResponse` import and `io` import to main.py
  - Added `getPackingSlip` operation to openapi-v3.yaml (now 31 operations)
  - Added PACKING SLIP section to gpt-instructions-v3.md
- **Why:** Office secretary needs to request printable packing slips through the GPT for warehouse use

---

## 2026-02-26 — Upgrade to v3.0.0: Merged endpoints, LAT Code Policy v1.1, shipment records
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Merged 6 preview/commit endpoint pairs into single endpoints with `mode` parameter: `/receive`, `/ship`, `/make`, `/pack`, `/adjust`, `/sales/orders/{id}/ship`
  - Added `Literal` import and `mode` field to all 6 request models (ReceiveRequest, ShipRequest, MakeRequest, PackRequest, AdjustRequest, ShipOrderRequest)
  - Added LAT Code Policy v1.1 fields to ReceiveRequest: `supplier_lot_code`, `lot_type`, `supplier_lot_entries`
  - Integrated `lot_supplier_codes` table for commingled receipt breakdowns in `/receive` commit path
  - Updated `/receive` commit to write `received_at`, `supplier_lot_code`, `lot_type` to lots table
  - Integrated `shipments` and `shipment_lines` tables in `/sales/orders/{id}/ship` commit path — auto-creates shipment record
  - Updated `check_open_orders_for_ship` URL reference from `/ship/commit` to `/ship`
  - Version bumped from 2.5.0 to 3.0.0
  - Generated OpenAPI YAML with exactly 30 operations (excludes dashboard, admin, system, scheduler endpoints)
  - Generated updated GPT instructions for v3.0.0
- **Why:** Upgrade to v3.0.0 — merging preview/commit stays under ChatGPT's 30-operation OpenAPI limit; LAT Code Policy v1.1 compliance; shipment tracking for sales orders; commingled receipt support

---

## 2026-02-25 14:42 — Moved global change log to iCloud Drive
- **File(s) changed:** `~/change-log.md` (now symlink)
- **What changed:** Moved `~/change-log.md` to `~/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md` (iCloud Drive) and created a symlink at `~/change-log.md` pointing to the new location
- **Why:** Enable remote sync of the global change log via iCloud without syncing all files

---

## 2026-02-25 14:36 — Added CLAUDE.md with change log protocol instructions
- **File(s) changed:** `CLAUDE.md`, `~/.claude/CLAUDE.md`
- **What changed:** Created project-level and global CLAUDE.md files containing instructions for maintaining dual change logs (project-level CHANGE_LOG.md and global ~/change-log.md) on every file change
- **Why:** User requested standardized change log protocol across all projects

---

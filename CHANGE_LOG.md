# Change Log

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

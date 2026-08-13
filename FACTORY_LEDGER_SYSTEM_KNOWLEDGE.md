# Factory Ledger System Knowledge

> **Purpose.** This is the authoritative reasoning guide for an AI or developer discussing the current Factory Ledger and its dashboards. It describes what the checked-in implementation actually does, distinguishes actual production from planning, and calls out places where the software, live data, and apparent business intent diverge.
>
> **Audit basis.** Repository `sevenwells72/factory-ledger`, `main` at commit `195fe68e7e59a955cceb24bbfccf1c9f9f8bcaa6` (matching `origin/main` when inspected on 2026-08-12, after merge `dc2ed6a` and the description-limit follow-up); production database inspected read-only at 2026-08-12 16:33 UTC; live health and integrity endpoints rechecked at approximately 2026-08-12 12:34 America/New_York. **Post-audit amendment (2026-08-13):** this document was amended after `195fe68` for the deployed Daily Entries endpoint, Activity-tab reorder and four-row expanders, migration-039 entry-time caveat, 79-test suite, and dashboard cache-version advances. Other point-in-time facts remain based on the stated audit unless explicitly amended. Point-in-time counts below will change. Code paths and table names are cited so future readers can revalidate them.
>
> **Evidence labels used here**
>
> - **CONFIRMED FROM CODE** — directly implemented in the cited current source.
> - **CONFIRMED LIVE** — observed in the production database or live endpoint at the audit time.
> - **STRONGLY INFERRED** — the repository strongly indicates an intent, but does not encode an authoritative business definition.
> - **UNCERTAIN / NEEDS OWNER CONFIRMATION** — material business meaning cannot be established from the repository.

# Factory Ledger in 2 Minutes

Factory Ledger is CNS Confectionery Products' operational inventory and production ledger. Operators or Custom GPT Actions record receipts, batch making, packing, shipments, and adjustments through a FastAPI API. Every inventory-changing transaction has one or more signed `transaction_lines`: positive pounds add inventory and negative pounds remove it. Lots tie those quantities to traceable physical inventory. Sales orders, customers, BOM/formula data, notes, scheduling configuration, and shipment support records surround that core ledger.

The authoritative fact that inventory moved is not a dashboard card, a schedule row, or a GPT statement. It is the current effective ledger: `ledger_current_transactions` plus `ledger_current_transaction_lines`, filtered to `effective_status = 'posted'`. Those views project append-only corrections over original `transactions` and `transaction_lines`. The API's canonical balance query is the `POSTED_LINES` SQL fragment in `main.py`. Actual production is a posted `make` or `pack` transaction with a positive output line. A schedule is only a plan.

The Dashboard is a read/presentation layer, not the ledger. It is a static HTML/CSS/JavaScript site on Netlify that calls the Railway API backed by Supabase Postgres. Its main Operations view shows finished goods, batches, ingredients, and a Production Calendar. Other views show shipping/receiving activity, notes, sales orders, a conceptual process-flow board, a Sankey visualization, traceability, and a completely separate browser-local production planner.

The main actual-production paths are:

```text
Physical activity / operator instruction
  -> Custom GPT Action or direct API request
  -> Railway FastAPI (`main.py`)
  -> Supabase Postgres ledger and support tables
  -> current/effective ledger views
  -> dashboard read APIs
  -> browser-side categorization, conversion, and display
```

The principal product categories are ingredients, packaging, intermediate batches, and finished products. Granola and coconut normally use `make` to turn formula ingredients into a batch, followed by `pack` to turn batch pounds into finished-product pounds/cases. The live Granola catalog now includes a separately identified, kosher-supervised Sunshine SS Classic #9 family (batch SKUs 90025/90026 and finished SKUs 70013–70018). A non-empty batch `verification_notes` value is surfaced as `production_warning` by formula lookup and `/make`, and the Floor GPT must relay it before commit. Graham cracker crumbs are different: SKU 31011 is a received 50-lb-bag ingredient and SKU 31012 is a packed 10-lb-case finished good. Current activity correctly uses `pack` at 1:1 pounds, but labels, box erection, packing stages, labor, scrap, and packaging consumption are not ledger events.

The biggest current limitations are:

- product family and calendar meaning are reconstructed from names and exact SKU lists in browser code, not stored as a stable production taxonomy;
- several legacy database views and API endpoints ignore effective corrections, while some correction fields are stored only inside JSON and are not honored by normal read paths;
- formulas, output sizes, product BOMs, line assignments, and dashboard configuration are separate mutable structures and can disagree;
- production warnings are advisory and narrowly surfaced: formula lookup and `/make` return them, but search, packing, dashboards, and direct backend commits do not enforce an acknowledgment;
- scheduling has two unrelated implementations—one database/API scheduler and one browser-local planner—and neither reconciles plan to actual activity;
- traceability is not a fully recursive, correction-aware chain from supplier lot through batch and pack to customer;
- the shared API credential is embedded in client/GPT configuration, CORS is unrestricted, and shared-key writes cannot establish an individual operator identity;
- the live 85/100 integrity score is a coarse checklist, not a proof of ledger correctness.

When totals disagree, always trace the chain `business process -> data capture -> ledger records -> effective-state query -> unit/category calculation -> UI`. Do not fix a missing category only in the UI if the workflow, product mapping, or ledger event is absent.

---

# Part 1 — Executive System Overview

## What the Factory Ledger is

Factory Ledger records the physical flow of pound-denominated inventory through a food manufacturing operation. It covers:

- raw-material and packaging receipts;
- formula-driven batch production (`make`);
- conversion of batch or bulk material into finished products (`pack`);
- standalone and sales-order shipments;
- found inventory and manual adjustments;
- lot identity, supplier-lot metadata, consumption, merging, and reassignment;
- customers, sales orders, shipment support records, notes, product catalog data, formulas/BOMs, and production-planning metadata;
- post-entry void/amend/restore corrections and day certifications introduced in Phase 1 of the ledger-integrity work.

The expected users are floor/fulfillment operators, sales/order users, management dashboard users, administrators maintaining products/BOMs, and Custom GPTs acting as conversational API clients. The role-specific Floor & Fulfillment GPT instructions identify Arturo as a primary bilingual floor user (`gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md`). There is no individual user account model in the API; all protected operations use one shared API key.

## Authoritative production record

**CONFIRMED FROM CODE:** A completed actual production event is a current, posted ledger transaction of type `make` or `pack`, with its output represented by a positive effective transaction line. The original rows are in `transactions` and `transaction_lines`; the effective projection after correction is in `ledger_current_transactions` and `ledger_current_transaction_lines` (migration `migrations/039_trial_timestamp_integrity.sql`, mirrored by `tests/schema/schema.sql`). `main.py` defines `POSTED_LINES` and uses it for canonical inventory balances.

This means:

- `make` means formula ingredients were consumed and a batch product was added;
- `pack` means source material was consumed and a target finished product was added;
- counts such as batches, pans, cases, and pallets are normally derived from positive pounds plus product metadata; they are not independent ledger quantities;
- a row in `production_schedule`, a card in the standalone planner, a sample fallback visualization, a note, or a GPT claim is not evidence that production occurred.

## Factory Ledger versus Dashboard

The Factory Ledger is the data and mutation system. The Dashboard is a consumer of selected API read models. The Dashboard can omit valid ledger records because it uses configured lists and name-based categories. Conversely, the conceptual process and scheduling views can display modeled or sample information that is not a ledger fact.

```text
FACTORY ACTIVITY
  Physical receipt, making, packing, shipment, count correction
        |
        v
FACTORY LEDGER
  Signed, lot-linked transaction lines + support records + corrections
        |
        v
READ MODELS AND CALCULATIONS
  effective views, API SQL, case/batch/pan conversions, categories
        |
        v
DASHBOARD
  operational visibility, investigation, order/readiness and planning aids
        |
        v
MANAGEMENT DECISIONS
  replenishment, production priority, fulfillment, trace/recall inquiry
```

## Relationship to other systems

- **Odoo:** `products.odoo_code` is the operational SKU identifier and many product names/configurations reflect Odoo. There is no live Odoo API integration in the repository. Factory Ledger's own database is the current source for inventory movements and its sales-order records.
- **Custom GPTs:** The main Factory Ledger GPT and the Floor & Fulfillment GPT call curated OpenAPI subsets of the backend. They are clients, not a source of truth. A mutation is factual only after the API commits it and returns a receipt.
- **Supabase:** Hosted Postgres database and therefore the durable operational data store.
- **Railway:** Hosts the FastAPI application.
- **Netlify:** Publishes the static dashboard (`netlify.toml`). The same dashboard directory is also mounted by FastAPI at `/dashboard`.
- **WhatsApp/export intake:** Phase 1 certifications allow `source_type = 'whatsapp_export'`, but no background ingestion service is implemented in this repository. Any such import must call the API or write records through another external process.

---

# Part 2 — System Architecture

## Actual architecture

```text
Operator / Sales User / Custom GPT
     |  HTTPS, JSON, X-API-Key for protected routes
     v
Railway: FastAPI 0.104 (`main.py`, Python 3.11)
     |  raw SQL through psycopg2 ThreadedConnectionPool (2..20)
     v
Supabase: PostgreSQL
     |  core rows + current/effective views
     +------------------------------+
     |                              |
     v                              v
Dashboard read endpoints       GPT Action endpoints
(`/dashboard/api/*`)           (main 30-op / floor 22-op schemas)
     |                              |
     v                              |
Netlify static HTML/CSS/JS <--------+
  main dashboard + activity + orders + notes
  process-flow + Sankey + traceability
  browser-local production board
```

## Technology and runtime

| Layer | Current implementation | Evidence |
|---|---|---|
| Frontend | Framework-free HTML, CSS, and JavaScript | `dashboard/index.html`, `dashboard/dashboard.js`, individual `.html` views |
| Backend | FastAPI 0.104.1, Pydantic request models, one large application module | `main.py`; `requirements.txt` |
| Database | Supabase-hosted PostgreSQL | `DATABASE_URL`; SQL schema/migrations |
| Access layer | Raw psycopg2 SQL; no ORM | `psycopg2-binary`, `ThreadedConnectionPool`, helpers in `main.py` |
| Files/export | `reportlab` PDF packing slips; `openpyxl` order matrix; CSV late-record export | `main.py:generate_packing_slip`, `export_orders_matrix`, `export_late_records_csv` |
| Static serving | Netlify publishes `dashboard/`; FastAPI also mounts the same directory | `netlify.toml`; `main.py` final `StaticFiles` mount |
| Authentication | Shared `X-API-Key`; packing slip also accepts a key query parameter | `main.py:verify_api_key`, `verify_api_key_query` |
| CORS | All origins, methods, and headers allowed | `main.py` CORS middleware |
| Python runtime | 3.11.7 | `runtime.txt` |

The backend declares version 3.1.1, the main OpenAPI Action schema declares 3.4.0, and the main GPT instructions declare 3.7.0. These numbers describe different artifacts and are currently not synchronized.

## Service behavior

- `main.py` fails startup if `DATABASE_URL` or `API_KEY` is missing, then creates a 2–20 connection pool.
- A startup migration block performs multiple `ALTER`/backfill operations for accumulated catalog and transaction changes. Individual migration failures are logged as nonfatal warnings, so application health does not prove every inline migration succeeded.
- There is no queue, worker, cron scheduler, or durable background-job framework in the repository. `scripts/daily-health-ping.sh` is an external health helper, not an application job.
- There is no server-side file store. PDF/XLSX/CSV artifacts are generated from database queries when requested. The standalone production board persists its state only in browser `localStorage`.
- Most business operations are synchronous SQL transactions. The `get_transaction()` helper commits or rolls back and returns connections to the pool.
- Read-only requests have a retry/tripwire mechanism covered by `tests/test_readonly_tripwire.py`.

## Authentication and exposure boundary

Protected routes depend on `verify_api_key`; `/sales/orders/{order_id}/packing-slip` may authenticate through a query key. Root and `/health`, all `/dashboard/api/*` reads, and `/audit/integrity` are public. Dashboard note mutations and all operational/admin mutations are protected.

**Important security fact:** the static browser code and generated GPT instructions contain the shared credential. This document intentionally does not reproduce it. Because the key is shared and the dependency returns a boolean rather than an operator identity, `_operator_id()` records `legacy-shared-key` rather than a person. CORS permits any origin. The system therefore has authorization-by-shared-secret, not user authentication, roles, or trustworthy per-operator attribution.

## API families

| Family | Examples | Auth | Role |
|---|---|---:|---|
| Health/public diagnostics | `/`, `/health`, `/audit/integrity` | No | Availability and coarse integrity |
| Catalog/inventory/lot | `/products/*`, `/inventory/*`, `/lots/*`, `/bom/*` | Yes | Resolution, balances, catalog and lot metadata |
| Ledger writes | `/receive`, `/ship`, `/make`, `/pack`, `/adjust`, `/void/*`, `/records/*` | Yes | Actual activity and corrections |
| Trace/history | `/trace/*`, `/transactions/history`, `/production/day-summary` | Yes | Operational investigation |
| Customers/orders | `/customers*`, `/sales/orders*`, `/sales/dashboard` | Yes | Sales and fulfillment |
| Main dashboard reads | `/dashboard/api/*` GET, including `/dashboard/api/activity/daily-entries` | No | Static dashboard data and effective daily-entry visibility |
| Notes writes | `/dashboard/api/notes*` non-GET | Yes | Dashboard annotations |
| Admin | `/admin/products`, `/admin/bom*`, `/admin/product-bom*`, `/admin/sql`, lot merge | Yes | Direct catalog/data maintenance |
| Scheduling | `/production/requirements`, `/schedule` | Yes | Planning, not actuals |

**CONFIRMED FROM CODE:** `GET /dashboard/api/activity/daily-entries` is a public dashboard read with required `date=YYYY-MM-DD` and optional `date_mode=event|entered` (default `event`). It reads `ledger_current_transactions` joined to `ledger_current_transaction_lines`, restricted to `effective_status='posted'`. It is deliberately absent from both GPT Action schemas; their operation counts remain 30 main and 22 Floor.

The root main GPT schema exposes exactly 30 operations (`openapi-gpt-v3.yaml`). The Floor schema exposes 22 (`gpt-configs/schemas/openapi-floor.yaml`). Many backend capabilities—corrections, certifications, scheduling, most admin tools, dashboard reads, found-inventory workflows—are not callable by the main GPT unless its Action schema changes. The main schema is already at the historical 30-operation cap documented in the repository.

The 2026-08-12 warning rollout changed descriptions/responses on existing operations only. Both schemas now tell the GPT that `getBatchFormula` and `/make` may return `production_warning`; operation counts remain 30 and 22. **CONFIRMED LIVE (owner-reported 2026-08-12):** both GPT editor Action configurations were refreshed from these schemas, and the Floor instruction configuration was refreshed from the rebuilt dist artifact.

Only the Floor instruction source gained the explicit MAKE rule to relay English and optional Spanish warning text verbatim before commit. Its rebuild also removed the redundant dispatch sentence “For order dispatch, also quote new order_status.” The main GPT instruction file did not change for this rollout; its new warning contract is documented in the updated main Action schema. **CONFIRMED FROM CODE:** the generated Floor artifact contains the new rule and no longer contains that dispatch sentence.

Two independent ChatGPT editor limits are now part of the deployment contract:

- the Floor instruction artifact must remain below 8,000 characters; current `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` is 7,833 Unicode characters, generated 2026-08-12 from the split sources;
- every OpenAPI `description`/`summary` must be at most 300 characters. The editor rejected the initial warning schema because the Floor `/make` description was 409 characters. Commit `195fe68` reduced the three offenders to 281 characters (Floor `/make` operation), 267 (Floor `OrderStatus`), and 279 (main `/pack` 200 response), with no path, operation ID, or structural changes.

The Floor schema separates sales-order dispatch into preview `shipOrder` and commit `commitShipOrder`, and its generated instructions require explicit approval plus a receipt before success. Both operations are nevertheless marked `x-openai-isConsequential: false`; safety therefore depends on prompt adherence and backend mode separation rather than the platform's consequential-action confirmation behavior. The main GPT uses a combined order-shipment operation rather than that dedicated two-operation contract.

## Configuration and environment

| Item | Purpose |
|---|---|
| `DATABASE_URL` | Required Postgres connection string |
| `API_KEY` | Required shared API credential |
| `dashboard/dashboard_config.json` | Exact product lists, grouping and pallet assumptions for main inventory panels |
| `openapi-gpt-v3.yaml` | Main GPT Action contract |
| `gpt-configs/schemas/openapi-floor.yaml` | Floor GPT Action contract |
| `gpt-instructions-v3.md` | Main GPT behavior rules |
| `gpt-configs/sources/*.md` + `build_gpt_instructions.py` | Source and builder for generated Floor instructions; enforces the 8,000-character instruction limit |
| `netlify.toml` | Static dashboard publish root and no-cache headers |
| `pytest.ini`, `tests/requirements-test.txt` | Database-backed test configuration |

## Repository structure

- `main.py` — backend, models, SQL, exports, dashboard API, scheduler, integrity endpoint, and startup migrations.
- `dashboard/` — primary UI and all standalone visualization/planning pages.
- `migrations/` — active migrations 037–040 and the Phase 1 dry run.
- `archive/migrations-applied/` — historical, already-applied migrations 003–036 and reconciliation SQL.
- `tests/` — schema snapshot and current Python/JavaScript regressions.
- `gpt-configs/` and root GPT files — current Action schemas and instructions.
- `scripts/` — health, schema dump, old void cleanup, test setup, and pack-format mapping tools.
- `audits/` — historical snapshots and audit reports. They illuminate evolution but must be rechecked against current code.
- `docs/` — deployment/incident records, including Phase 1 integrity deployment.
- `archive/superseded-*` — explicitly non-current API schemas and prompts.

---

# Part 3 — Database and Data Model

## Model conventions

The schema snapshot is `tests/schema/schema.sql`; current changes are in `migrations/037_sales_order_flags.sql` through `040_product_pack_format.sql`. The snapshot was refreshed from the Phase 1/calendar mainline before the 2026-08-12 warning merge. **CONFIRMED FROM CODE:** it includes migration 039's append-only functions/triggers, `ledger_current_*` views and certifications, plus migration 040's `products.pack_format` column/check. It is therefore current for those structures; the eight SS Classic #9 catalog rows are production data and correctly do not appear in a schema-only dump. The live database remains more authoritative than a schema snapshot for current data and deployment state.

Inventory is represented in pounds even when the operator thinks in cases, bags, boxes, pans, or pallets. A `transaction_line.quantity_lb` is signed:

```text
positive = inventory enters a product/lot
negative = inventory leaves a product/lot
balance  = SUM(effective posted quantity_lb)
```

`created_at` is the database receipt time after Phase 1. `transactions.timestamp` is a legacy naive event timestamp. `occurred_at` is the timezone-aware event time introduced by Phase 1; `business_date` is the plant business date. The application has not fully switched reads to those new fields.

## Core ledger entities

### `products`

Purpose: one catalog row per operational item/SKU. Primary key `id`; `odoo_code` and `name` are unique identifiers when present.

Important fields:

- classification: `type` is exactly `ingredient`, `packaging`, `batch`, or `finished`;
- identity: `name`, `odoo_code`, `brand`, `product_category`, optional customer linkage/name;
- units: `uom`, `case_size_lb`, `pack_size_lbs`, `units_per_case`, `default_case_weight`;
- batch/retail behavior: `default_batch_lb`, `yield_multiplier`, historical yield fields, retail bag fields;
- workflow: `parent_batch_product_id`, `has_bom`, `bom_status`, `no_production`, `is_service`, `is_copack`;
- labeling: `label_type`; the code distinguishes at least house/private-label behavior in safeguards;
- calendar display: `pack_format` is nullable or one of `10lb`, `25lb`, `bagged`;
- governance: `active`, verification status/notes (English plus optional Spanish), storage/shelf-life/certification-related metadata. Non-empty batch `verification_notes` now become an advisory `production_warning` in `/make` and formula responses; they are still mutable product metadata, not a ledger event or enforced authorization.

Relationships: referenced by lots, every transaction line, formula and BOM rows, order lines, line assignments, schedules, allergens, notes, and verification history. `parent_batch_product_id` self-links a finished product to a source batch where configured.

Creation/modification: quick-create and found-with-new-product routes, product verification, and admin updates in `main.py`; startup migrations, catalog migrations, and approved direct production SQL can also mutate products. There is no append-only product version. Current live count: 209 products.

### `lots`

Purpose: physical or logical inventory lot for exactly one product. Primary key `id`; unique `(product_id, lot_code)` means the same textual lot code may exist for different products.

Fields include `product_id`, `lot_code`, `supplier_lot_code`, `received_at`, `entry_source`, `lot_type`, found-inventory location/age, status (`active` or `merged`), and self-referential merge metadata. A lot may be auto-generated, operator-specified, received, produced, packed, found, renamed, reassigned, or merged.

Transactions reference lots through lines. `ingredient_lot_consumption` separately records which lots production consumed. Current live count: 962 lots.

### `transactions`

Purpose: header for one business event. Primary key `id`. `type` is used as `receive`, `ship`, `make`, `pack`, or `adjust`; the database field itself is not constrained to that enum.

Important fields: legacy `timestamp`, `occurred_at`, `business_date`, `type`, raw `status`, notes, BOL/shipper, cases/case size, customer/order reference, adjustment reason metadata, `operator_id`, `created_at`, and `created_at_source`.

The Phase 1 database trigger makes original transaction rows append-only and forces database `created_at`. Raw historical `status='voided'` still exists, while new voids are corrections. Live at audit: 1,803 transactions; 1,769 effective posted and 34 effective voided.

### `transaction_lines`

Purpose: signed quantity movement for a transaction, product, and lot. Primary key `id`; FKs to `transactions`, `products`, and optional `lots`. `quantity_lb` is numeric and signed. Original rows are append-only under Phase 1.

Every valid inventory balance should derive from effective posted lines, not from mutable stock fields. Live count: 4,817 lines.

### `ingredient_lot_consumption`

Purpose: positive-magnitude trace support linking a `make` or `pack` transaction to each consumed ingredient/source product and lot. Primary key `id`; FKs to transaction, ingredient product, and ingredient lot; `quantity_lb` is positive.

It duplicates information that can often be inferred from negative transaction lines, but is the principal consumption table used by trace queries and the integrity check. It has no effective/correction view. Lot merge directly updates it, and lot reassignment currently does not update it. Live count: 2,789.

## Effective-state and audit entities

### `ledger_corrections`

Append-only correction event. UUID PK. Targets `transactions` or `transaction_lines`; event type is `amend`, `void`, or `restore`; stores `previous_values` and `replacement_values` JSONB plus non-empty reason, operator, and database timestamp. Database triggers block update/delete and override supplied creation time.

Live state contained one transaction void correction. Older voids remain raw status changes.

### `ledger_current_transactions`

View that chooses the latest correction for each transaction and exposes `effective_status` plus an `effective_record` JSON merge. **Caveat:** ordinary columns such as `occurred_at`, `business_date`, and `timestamp` remain the original columns. An amendment to those values exists in `effective_record`, but most application queries still read the original columns.

### `ledger_current_transaction_lines`

View that overlays the latest line correction on `product_id`, `lot_id`, and `quantity_lb`. It is the correct line-level source after lot merge/reassignment or an amendment.

### `certifications` and `current_certifications`

Append-only daily cutoff/certification records with UUID, `business_date`, `certified_at`, operator, source type (`manual` or `whatsapp_export`), notes, and a superseding correction chain. The current view chooses the latest certification per business date. `/records/late` compares post-certification rows and corrections to the certified cutoff.

Live state had only two smoke-test rows for 1900-01-01 (one original and one superseding correction), so no real operating day was certified.

## Formula and product-structure entities

### `batch_formulas`

The active execution recipe used by `/make`. Each row links batch `product_id` to `ingredient_product_id`, with pounds per batch and `exclude_from_inventory`. There is no version, effective date, or active flag. Live: 181 rows covering all 25 active batch products. The 15 new rows are exact copies for 90025 from 90002 (seven rows) and 90026 from 90001 (eight rows).

`exclude_from_inventory` is used for water or other deliberately non-stocked components. Those pounds remain in the conceptual formula but no negative ledger line is created.

### `product_bom`

Finished-product-to-component mapping used by product admin, production requirements, scheduling, and pack source resolution. Each row links `finished_product_id` to `component_product_id` and a quantity. It is not enforced by `/pack`; its packaging rows do not cause packaging consumption. Live: 114 rows covering 38 finished products; 34 active finished products had no row. The six new SS Classic #9 finished goods use `parent_batch_product_id` only and have no `product_bom` rows.

### `boms` and `bom_lines`

An older/spec-oriented BOM model with headers and lines. Live counts were both zero and current execution code does not use them. Do not treat them as the production recipe.

### Allergens and product verification

`allergens` and `product_allergens` attach allergen metadata. `product_verification_history` records catalog verification/rename/merge actions. These are governance/support data, not quantity movements.

## Customers, orders, and shipping

### `customers` and `customer_aliases`

Customers have an integer PK, name/contact/address fields, active state, and timestamps. Aliases support resolution. The API can auto-create customers during order entry, which can also create typo duplicates. Live: 75 customers, 74 active; five aliases.

### `sales_orders`

Order header keyed by integer `id`, with unique `so_number`, customer, requested ship date, status, addresses/references, notes, and timestamps. Current live statuses were 160 shipped, 60 cancelled, and 16 confirmed. The supported state machine is described in Part 5.

### `sales_order_lines`

Order product, ordered pounds and/or units, case information, shipped pounds, status, and service-item fields. Quantity can be entered as pounds, cases, bags, or boxes; API code normalizes physical demand to pounds. Service lines can have zero weight.

### `sales_order_shipments`

Associates an order to shipment/transaction facts and supports order fulfillment history. Live count: 414.

### `shipments` and `shipment_lines`

Shipping support model alongside ledger ship transactions. `shipments` is a shipment header; `shipment_lines` associates shipped product/quantity and transaction. Standalone `/ship` now creates these records. Order shipment creates support records as well, but the header/line/order tables overlap conceptually. The actual inventory decrease remains the negative effective ledger line.

Live: 285 `shipments` and 556 `shipment_lines`. One current posted ship transaction after the integrity cutoff lacked a shipment line.

### `sales_order_flags`

Dashboard-only annotation state keyed by `so_number`, introduced by migration 037. It is separate from canonical `sales_orders.status` and has no foreign key. A “ready” UI flag must not be mistaken for the order state machine.

## Scheduling and capacity

### `production_lines`

Four live rows: Granola Baking (`granola`), Coconut Sweetened (`coconut`), Bulk Packing (`bulk_pack`), and Pouch Line (`pouch`). Fields are integer `id`, name, unique code, and active.

### `line_capacity_modes`

Alternative capacities per production line: worker requirement and one of batches/day, pallets/day, or bags/day, with optional pack size and default flag. Live values are detailed in Part 12.

### `product_line_assignments`

Many-to-many product-to-line eligibility. Live: 63 assignments—18 batch products to granola, four batch products to coconut, 32 finished products to bulk packing, and nine finished products to pouch.

### `production_schedule`

Persisted plan rows: date, line, product, planned batches/pounds/bags, assigned workers, status, linked order numbers, overproduction, notes, and timestamps. Allowed statuses are `planned`, `confirmed`, `in_progress`, `completed`, and `cancelled`. Live table was empty. There is no FK from actual ledger transactions to a schedule row.

### `scheduling_config`

Key/value configuration used by the API scheduler for workforce, horizon, Friday capacity, and related assumptions. It is separate from the embedded settings of the standalone browser planner.

## Notes, adjustments, and support tables

- `notes`: dashboard notes/reminders/todos with type, completion state, dates, and related product/customer/order fields. Live: 18.
- `inventory_adjustments`: support/audit metadata written best-effort by found-inventory and adjustment workflows. It is not the balance source. Live: 130.
- `lot_reassignments`: audit log for product reassignment of a lot; live count zero. Effective line corrections, not this table, determine the corrected ledger product.
- `lot_supplier_codes`: supports more than one supplier lot for a commingled system lot/receipt.
- `adjustment_reason_codes` and `reassignment_reason_codes`: validate/catalog adjustment, found-inventory, and reassignment explanations.
- `oauth_tokens`: present but not used by the current API integration model.
- `_backup_20260305_products`, `_backup_20260305_lots`, `_backup_20260305_transactions`, `_backup_20260305_transaction_lines`, `_backup_20260305_batch_formulas`, `_backup_20260305_ingredient_lot_consumption`, `_backup_20260305_sales_order_lines`, `_backup_20260305_shipments`, and `_backup_20260305_shipment_lines`: one-off snapshots retained in the schema. No current application query consumes them. They are recovery/history artifacts, not live truth.

## Legacy views and duplicated meanings

Views such as `inventory_summary`, `lot_balances`, `low_stock_alerts`, `production_history`, `todays_transactions`, and `v_lot_quantities` query raw transaction tables. Some use obsolete values (`products.type='finished_good'`, `transactions.type='production'`) that cannot match current catalog/application enums. Five legacy `/dashboard/*` endpoints consume these views. They are not the main static dashboard APIs and are not correction-aware.

Duplicated concepts that require care:

- raw transaction `status` versus correction-derived `effective_status`;
- legacy `timestamp` versus `occurred_at`, `business_date`, and immutable `created_at`;
- signed negative transaction lines versus positive `ingredient_lot_consumption` rows;
- `batch_formulas`, `product_bom`, and empty `boms`/`bom_lines`;
- ledger `ship` transactions, `shipments`, `shipment_lines`, and `sales_order_shipments`;
- `sales_orders.status` versus `sales_order_flags`;
- database `production_schedule` versus the browser-local production board;
- `default_batch_lb`, formula sum, yield multiplier, and display-specific “pan” counts.

## Point-in-time live inventory-model snapshot

At the refreshed audit time there were 209 products: 25 batch (all active), 75 finished (70 active), 77 ingredients (all active), and 32 packaging items (all active). Nine finished products were marked copack; 26 products were marked `no_production` by migration 038, including Graham 31011. `pack_format` was populated for 28 products (4 `10lb`, 15 `25lb`, 9 `bagged`; 181 null). There were 962 lots and 1,803 transactions spanning 2026-01-28 through 2026-08-11; the eight new catalog records had not changed those ledger counts. Effective state remained 1,769 posted and 34 voided transactions.

---

# Part 4 — Product Model

## Actual hierarchy

There is no normalized `product_family`, `variant`, `label`, or `packaging_format` hierarchy. The actual persistent model is flatter:

```text
products row
  identity: id + Odoo SKU (`odoo_code`) + unique name
  class: ingredient | packaging | batch | finished
  optional source: parent_batch_product_id and/or product_bom rows
  operational traits: case size, batch size, yield, copack/service/no-production
  presentation traits: brand, label_type, pack_format, name text
  optional customer association
```

Families and variants are mostly encoded in names and curated frontend lists. `dashboard_config.json` selects exact names for Finished Goods, Batches, and Ingredient categories. `dashboard.js` uses name heuristics to classify calendar production as granola, coconut, or Graham. This is presentation taxonomy, not a database-enforced product hierarchy.

## Product identity and resolution

The API resolves a user phrase in tiers: exact matches, word-order-independent matching, then trigram similarity (`main.py` resolution helpers and `/products/search`, `/products/resolve`). Operators/GPTs are instructed to search first and pass a resolved catalog name into transactions. `odoo_code` is the durable business SKU, but many internal queries and dashboard lists still depend on exact `name` strings.

## Granola

Granola spans three model levels:

1. ingredients such as oats, nuts, sweeteners, fruit, and chocolate;
2. `batch` products with SKUs largely in the 900xx/950xx range and one `batch_formulas` recipe each;
3. `finished` products in bulk, 10-lb, 25-lb, and retail/bagged case formats.

The relationship from a finished granola SKU to a batch can be expressed through `parent_batch_product_id` and/or `product_bom`; neither is a complete enforced hierarchy. A normal workflow is `/make` ingredients into a batch lot, then `/pack` batch pounds into the customer/brand/format finished SKU. Private-label identity is a separate finished product and often exists mainly in the name, brand/label fields, and frontend grouping.

The Production Calendar counts positive `make` pounds divided by `default_batch_lb * yield_multiplier` as granola batches. Packed granola is split by nullable `pack_format` into 10-lb, 25-lb, and bagged groups. Products that are not recognized by the family/name heuristic or do not have a mapped pack format can be absent from the calendar summary even though their ledger event exists.

### Sunshine SS Classic #9 — kosher-supervised family

**CONFIRMED LIVE at 2026-08-12 16:33 UTC:** eight active Sunshine, house-label catalog rows were inserted directly in one approved production SQL transaction. They are distinct SKUs, not renamed aliases of the ordinary Classic #9 family.

| ID | SKU | Product | Type / relationship | Size / format |
|---:|---:|---|---|---|
| 283 | 90025 | Batch SS Classic Granola #9 (Kosher Ignition) | `batch`; formula copied exactly from 90002 | 323-lb batch |
| 284 | 90026 | Batch SS Classic Chocolate Chip Granola #9 (Kosher Ignition) | `batch`; formula copied exactly from 90001 | 348-lb batch |
| 285 | 70013 | Granola SS Classic #9 Bulk per/lb | `finished`; parent 90025 | no case size; `pack_format=NULL` |
| 286 | 70014 | Granola SS Classic #9 25 LB | `finished`; parent 90025 | 25 lb; `pack_format=25lb` |
| 287 | 70015 | Granola SS Classic #9 10 LB | `finished`; parent 90025 | 10 lb; `pack_format=10lb` |
| 288 | 70016 | Granola SS Classic Chocolate Chip #9 Bulk per/lb | `finished`; parent 90026 | no case size; `pack_format=NULL` |
| 289 | 70017 | Granola SS Classic Chocolate Chip #9 25 LB | `finished`; parent 90026 | 25 lb; `pack_format=25lb` |
| 290 | 70018 | Granola SS Classic Chocolate Chip #9 10 LB | `finished`; parent 90026 | 10 lb; `pack_format=10lb` |

Every row contains the English `verification_notes` requirement: the oven must be turned on by the owner (Blubber) or his designated messenger, and production must not start without that ignition. The live `verification_notes_es` fields are empty. The requirement is master-data guidance; it does not record who ignited the oven or prove compliance for a run.

The two new formulas are ingredient-for-ingredient and quantity-for-quantity copies: 90025 has the seven rows from 90002, while 90026 has the eight rows from 90001. The six finished products use `parent_batch_product_id`; none has `product_bom` rows or a production-line assignment. The two batch products also lack line assignments. Consequently `/make` and `/pack` can operate through formula/parent data, and the make response can offer the related finished SKUs, but the API scheduler cannot plan this family correctly from its `product_bom`/line model.

The exact new names are not present in `dashboard_config.json`, so the main Finished Goods and Batch Inventory panels omit them. If actual events are posted, the Production Calendar will classify their names as Granola; 10-/25-lb packed variants have usable `pack_format`, while the two bulk per-pound variants have null format and will be omitted from the packed case buckets.

## Coconut

Coconut likewise has raw desiccated coconut ingredients, sweetened batch intermediates, and finished formats/labels. Batch SKUs distinguish Fancy, Flake, Medium, and Toasted. Finished names distinguish pack weights and brand/customer labels such as CNS or foodservice/private-label variants.

Sweetened coconut formulas include water with `exclude_from_inventory=true`. `yield_multiplier=1.11` hydrates the declared output beyond `default_batch_lb`. The calendar calls the derived make unit a “pan”: positive output divided by `default_batch_lb * yield_multiplier`. That UI unit is not stored as a pan record.

Toasted coconut is modeled as a batch, but the exact real-world boundary between sweetening, cooling, toasting, and a “pan” is not represented by stage events. The standalone planner models those stages more explicitly, but only as planning assumptions.

## Labels, customers, and packaging

Label/customer-specific versions generally exist as distinct finished `products` rows. `label_type`, brand, names, and optional customer fields provide metadata, but there is no production-run label-application record. Packaging products exist and may appear in `product_bom`, yet `/pack` does not consume `product_bom` packaging. A finished case therefore proves pounds were packed into that SKU; it does not prove a particular label roll or box lot was consumed.

## Graham cracker crumbs

The current catalog distinguishes the two required products:

| SKU | Product | Type | Operational meaning | Live balance at audit |
|---|---|---|---|---:|
| 31011 | Graham Cracker Crumbs – 50 LB | `ingredient` | received bulk/bagged crumb; sold as-is or used as pack source; `uom='50 lb bag'`; `case_size_lb=50`; `no_production=true` | 29,990 lb |
| 31012 | Graham Cracker Crumbs – 10 LB | `finished` | CNS house-label, copack finished case; `uom='10 lb case'`; `case_size_lb=10`; `no_production=false` | 4,200 lb |

`product_bom` for 31012 contains 10 lb of 31011, one clear #10 bag (SKU 21003), and one Box 24 (SKU 21011). Only 31012 is assigned to Bulk Packing. There is no `parent_batch_product_id` between them.

The actual 10-lb workflow is recorded as `pack`: negative pounds from 31011 lot(s), positive pounds into a 31012 lot, normally 1:1. At the audit time the two products had six receive transactions, 23 pack transactions, 60 ship transactions, and seven adjustments in their combined history. The 2026-08-10 calendar activity included 4,200 lb / 420 cases of packed 31012.

What is **not** modeled as ledger data:

- opening/dumping individual 50-lb bags;
- case erection or box-preparation steps;
- label selection/application or label-lot consumption;
- an operator, workstation stage, start/end time, downtime, or labor count for the run;
- packaging consumption from the listed bag and box BOM rows;
- scrap, giveaway, or case-by-case reconciliation.

Therefore the ledger proves bulk crumb pounds were converted into a finished 10-lb SKU; it does not prove every physical substep or packaging material.

## Catalog integrity observations

- Four active finished SKUs (70004, 70006, and new bulk-per-pound 70013/70016) have no `case_size_lb`; the integrity endpoint passes its case-size check because it only tests products that already have positive pack history.
- Thirty-four active finished products have no `product_bom`; the increase from 28 is the six new parent-linked SS Classic #9 finished goods.
- All 25 active batch products have formulas and default batch sizes, but formula sum and declared output can diverge materially. For 90008 Granola Fruit Nut Batch, current formula rows total 25 lb while `default_batch_lb` is 384.52. Live history shows older makes outputting 25 lb and a 2026-06-29 make outputting 769.04 lb while consuming 50 lb, proving catalog change/logic drift can create physical mass inconsistencies.
- `pack_format` is primarily a display classification and is sparse. Migration 040 adds the field/constraint; `scripts/propose_pack_format_backfill.py` performs a guarded exact-SKU mapping rather than the migration itself.

---

# Part 5 — Factory Ledger Write Path

## Common write envelope

All normal mutations enter through FastAPI in `main.py`. The conversational clients are instructed to resolve products first, preview an operation, obtain any required operator confirmation, then commit. The named `/receive`, `/ship`, `/make`, `/pack`, and `/adjust` endpoints themselves accept a `mode`; convenience `/preview` and `/commit` paths delegate to the same functions. `write_response_envelope` normalizes write responses, and `run_idempotent_write_with_readonly_retry` protects selected writes from a stale/read-only database connection. There is not yet a general idempotency-key contract; `IDEMPOTENCY_KEY_PLAN.md` is a plan.

Common data-entry facts:

- protected requests carry the shared API key, not a user session;
- the GPT is an API client, so it must never claim a mutation succeeded from conversational intent alone;
- `resolve_product_id`, `resolve_product_full`, `_tiered_product_search`, and `_resolve_single_product` turn an operator phrase into a catalog row;
- `find_or_create_lot` enforces the product/lot uniqueness rule;
- `generate_lot_code` constructs system lot codes from plant date, shipper abbreviation, and sequence;
- preview is a read/validation step; it is not a ledger event;
- a commit normally performs the header, lines, consumption/support rows, and order changes in one SQL transaction, except explicitly best-effort support logs.

## Receive: inbound ingredient, packaging, or finished inventory

**Entry point:** `main.py:receive` and the `/receive/preview` / `/receive/commit` aliases.

```text
Operator gives product, cases, case size, shipper, BOL, supplier lot
  -> product resolution
  -> total_lb = cases * case_size_lb
  -> preview returns proposed system/supplier lot and quantity
  -> commit finds/creates lot
  -> INSERT transactions(type='receive')
  -> INSERT transaction_lines(+total_lb)
  -> update lot receipt/supplier metadata
  -> inventory and receipt dashboard can read the event
```

Required operational fields include product, count/size, shipper, BOL reference, and supplier lot under current GPT instructions. The backend accepts a physical/system lot override; otherwise it generates one. Supplier lot falls back through supplied supplier-lot/lot values and ultimately `N/A` behavior. Multiple supplier codes for a commingled receipt can be stored in the support mapping used by the endpoint.

**Validation/derived values:** positive cases and case size; resolvable product; `total_lb = cases * case_size_lb`; lot uniqueness per product. **Gap:** the GPT instructions require the same supplier lot received on a different day to receive a new system lot, but the backend does not enforce that temporal rule by supplier-lot identity. A direct client can reuse an existing product/system-lot combination.

## Standalone ship: inventory not fulfilled through an order

**Entry point:** `main.py:ship` and preview/commit aliases.

```text
Customer + product + pounds/cases
  -> resolve/optionally create customer
  -> detect open sales orders for customer
  -> block unless order workflow is used or force_standalone=true
  -> lock eligible lots and calculate effective on-hand
  -> allocate FIFO or honor a pinned lot
  -> INSERT one ship transaction
  -> INSERT negative transaction line(s), one per allocated lot
  -> INSERT shipment + shipment_lines support records
  -> return transaction/shipment receipt
```

`validate_lot_deduction` and effective `POSTED_LINES` balances protect against an ordinary overdraw. A forced standalone override is recorded in notes. `order_reference` can be placed on the transaction but does not turn the shipment into sales-order fulfillment.

The modern standalone path now creates `shipments` and `shipment_lines`; older audit documentation that says it never does is stale. A `shipment_line` records product/quantity/transaction but not the allocated lot. Lot truth remains in `transaction_lines`.

## Make: formula ingredients to batch product

**Entry point:** `main.py:make`; formula lookup through `batch_formulas`; related helpers `get_sibling_skus` and lot allocation.

```text
Operator selects batch product, number of batches, output lot
  -> load product.default_batch_lb and yield_multiplier
  -> output_lb = default_batch_lb * batches * yield_multiplier
  -> load each batch_formulas row
  -> required component lb = formula.quantity_lb * batches
  -> skip stock deduction for excluded component or explicit manual exclusion
  -> allocate required lots FIFO or use overrides
  -> INSERT make transaction
  -> INSERT positive batch-output line
  -> INSERT negative ingredient lines
  -> INSERT positive-magnitude ingredient_lot_consumption rows
```

Preview reports inventory requirements, shortages, sibling-SKU ambiguity, excluded ingredients, output, and a `pack_needed` prompt if product relationships imply a finished pack step. Commit locks lots before deductions.

`resolve_product_full` now selects `verification_notes` and `verification_notes_es`. `build_production_warning(product)` trims the English note and, when non-empty, adds this advisory shape to both preview and successful commit responses:

```text
production_warning:
  verification_notes: <English note verbatim>
  verification_notes_es: <optional Spanish note verbatim>
  message: "PRODUCTION WARNING — relay to operator verbatim before proceeding: ..."
```

Preview also appends the English note to `preview_message`. Whitespace-only notes produce no warning. The rebuilt Floor instructions require the GPT to relay the English and Spanish notes verbatim when present and obtain operator confirmation before commit. **Important boundary:** `MakeRequest` has no warning-acknowledgment field and the backend does not block a direct commit. A compliant GPT sees the warning during preview; the warning returned after commit is evidence/visibility but is too late to prevent the run.

**Critical semantics:** output does not equal the sum of consumed formula rows. It is independently calculated from `default_batch_lb * yield_multiplier`. Excluded water or another component creates no negative stock line and does not reduce output. The endpoint does not enforce that the selected output is `type='batch'`, that a formula exists, or that formula mass reconciles with declared output. Current live catalog happens to give all active batch products a size and formula, but SKU 90008 demonstrates that later catalog edits can make the same endpoint produce dramatically different mass for the same nominal operation.

### Granola baking

The operator records one or more granola batch products through `/make`. Ingredient lots are selected FIFO/overridden, negative lines and ILC rows are written, and the positive batch line is the actual. The calendar later reconstructs “batches” by dividing positive pounds by the product's current declared batch output. It does not store ovens, pans, bake stage, individual mix number, workers, temperature, start/end time, or downtime.

For 90025/90026, the preview now carries the Kosher Ignition production warning before the same make mechanics run. The software still does not capture an ignition event, the responsible owner/messenger, or a signed acknowledgment. The control is currently product metadata + API warning + GPT instruction, not a ledger-enforced prerequisite.

### Coconut production

Coconut sweetening/toasting is also `/make`. Sweetened formula water is excluded from inventory, while `yield_multiplier` increases declared output. The calendar labels the derived count “pans,” even though the ledger stores pounds and a number of requested batches. The standalone planner's coconut cycles and next-day toasted locks do not write this transaction.

## Pack: source material to target finished SKU

**Entry point:** `main.py:pack`; add-in resolver `resolve_pack_add_ins`.

```text
Operator selects source product/lot, target product, cases, case weight
  -> total_lb = cases * case_weight_lb
  -> source requirement = total_lb (1:1 by pounds)
  -> validate/lock source lot inventory; warn on suspicious mapping
  -> optionally derive non-base add-ins from source batch formula
  -> INSERT pack transaction
  -> INSERT negative source line(s)
  -> INSERT positive target finished-product line
  -> INSERT ILC rows for source and any formula add-ins
```

If an output lot is omitted, the target can inherit the first source lot code. The endpoint does not require `source.type='batch'` or `target.type='finished'`, and does not enforce a `product_bom` relationship. It can therefore convert arbitrary products if a client submits them.

If `target.parent_batch_product_id` differs from the selected source, `resolve_pack_add_ins` may load the parent batch formula, treat the selected source as the base component, scale other non-excluded formula components to packed pounds, and consume them. This is separate from `product_bom`; packaging rows in `product_bom` are ignored.

### Granola packing

Batch product pounds become a label/format-specific finished SKU. Cases are input and stored as header metadata, but inventory truth is pounds. Bags/boxes/labels are not ordinarily deducted unless represented as formula add-ins—an unsuitable workaround for normal packaging BOM consumption.

### Graham cracker crumb packing

The operator packs source 31011 to target 31012. Ten-pound cases derive `total_lb = cases * 10`; source and target pounds are 1:1. This correctly creates an actual `pack` event and therefore can appear in the current Production Calendar. It does not consume the clear bags or Box 24 listed in `product_bom` and does not record label/case-preparation stages.

### Case/box preparation

**CONFIRMED FROM CODE:** no case-erection or box-preparation event/table/API exists. Packaging products can be received, adjusted, and shipped like inventory, and can be listed on `product_bom`, but `/pack` does not consume that BOM. The desired physical activity is therefore not represented as a separate actual, duration, work-center event, or reliable packaging deduction.

### Cleaning and non-production activities

There is no cleaning, sanitation, maintenance, downtime, shift, or labor-event entity. Such information could be put in free-text notes, dashboard notes, or scheduler notes, but those do not become production events. The process-flow board has conceptual stages and hardcoded worker numbers; it is not a cleaning/activity ledger.

## Adjust and found inventory

**Entry point:** `main.py:adjust` writes an `adjust` transaction and one signed line. Preview warns about a projected negative balance, but a negative escape-hatch adjustment is supported for reconciliation. `check_private_label_merge` blocks negative adjustments whose reason contains merge/deprecate/consolidate/migrate keywords for private-label products; it does not block all private-label decreases.

`add_found_inventory` finds/creates a `FOUND` lot, writes a positive adjust transaction/line, and best-effort writes `inventory_adjustments`. Its response does not expose the transaction ID consistently. `add_found_inventory_with_new_product` creates an unverified product plus ledger activity but does not write the same support audit row. `get_found_inventory_queue` feeds later verification.

## Corrections and later edits

### Void/amend/restore

`void_transaction` requires a reason and appends a `ledger_corrections` event. It does not insert reversal lines. `correct_transaction` appends an amendment for supported header values. `_append_transaction_line_correction` is an internal line-level primitive used by lot operations. Original transaction and line rows remain unchanged.

**Major limitation:** normal reads do not consistently consume header values from `effective_record`. An amendment to `occurred_at` or `business_date` can be recorded without changing the date used by history, calendar, or activity queries. A new Phase 1 void affects current inventory because `effective_status` is honored by canonical balance paths.

### Lot reassignment

`reassign_lot` changes `lots.product_id`, appends corrections changing the product on historical transaction lines, and best-effort logs `lot_reassignments`. It counts related `ingredient_lot_consumption` records but does not update them. The response's `production_usage_updated` wording reflects a count, not an actual ILC rewrite. Trace and effective ledger can disagree after reassignment.

### Lot merge

`merge_lots` appends line corrections that move effective line lot IDs to a surviving lot, directly updates ILC lot IDs, and marks/mutates lot merge metadata. This preserves the original line but not the original ILC value, so the audit model is mixed.

### Catalog, formula, and order edits

Admin product, `batch_formulas`, and `product_bom` routes update/delete rows directly. There is no formula/catalog version effective at production time. Consequently a historical display that divides old output by today's `default_batch_lb` can change meaning after a catalog edit.

Customer/order header and line routes also update state directly rather than through the ledger-correction mechanism. That is appropriate for mutable workflow state, but downstream shipment void behavior must update those support aggregates explicitly and currently does not.

## Sales-order entry and fulfillment

### Order creation

`create_sales_order` resolves/auto-creates a customer, resolves products, converts cases/bags/boxes/pounds to normalized quantity, creates a confirmed order and its lines, and returns warnings for implicit-pound or unusually low physical quantities. Services are permitted with zero weight. Automatic customer creation can turn an address/name typo into a duplicate; `FOLLOWUPS.md` records this concern.

Manual state transitions in `update_order_status` are:

```text
new -> confirmed | cancelled
confirmed -> in_production | cancelled
in_production -> ready | cancelled
ready -> in_production | cancelled
partial_ship -> cancelled
shipped -> invoiced
invoiced, cancelled -> terminal
```

Creation normally starts confirmed, so `new` is largely legacy. Shipment logic automatically sets `partial_ship` or `shipped`.

### Order shipment

`ship_order` previews or commits selected/all quantities. Commit:

1. locks and re-reads the order;
2. skips inventory for service lines;
3. creates one shipment header;
4. creates one ledger `ship` transaction per physical order line;
5. allocates effective inventory FIFO and writes negative lot lines;
6. writes `sales_order_shipments` and `shipment_lines`;
7. directly increments `sales_order_lines.quantity_shipped_lb` and line status;
8. auto-fulfills service lines;
9. sets order to `shipped` only when every line is fully fulfilled, otherwise `partial_ship`.

An all-service request cannot commit because the workflow requires at least one physical quantity and rolls back. A later Phase 1 void of one generated ship transaction restores effective inventory only; it does not decrement `quantity_shipped_lb`, remove/void shipment support rows, reopen line status, or change the order header. This is a high-impact reconciliation gap.

## Schedule versus actual at write time

`POST /schedule` with action `confirm` writes/upserts `production_schedule`. It never creates `make` or `pack` transactions. Conversely, committing `/make` or `/pack` never finds or completes a schedule row. The browser production board writes only `localStorage`. Thus:

```text
schedule suggestion/confirmation != actual production
actual make/pack != schedule completion
```

---

# Part 6 — Factory Ledger Read Path

## Canonical balance read

`main.py` defines `POSTED_LINES` as effective lines joined to current transactions with `effective_status='posted'`. `lot_on_hand`, `validate_lot_deduction`, `/inventory/current`, `/inventory/lookup`, lot-detail paths, the main dashboard inventory endpoints, order shipment availability, and the API scheduler inventory loaders use this pattern or the current views directly.

```text
ledger_current_transactions (void/restore state)
  JOIN ledger_current_transaction_lines (corrected product/lot/quantity)
  WHERE effective_status = posted
  GROUP BY product and/or lot
  SUM(quantity_lb)
```

This is the preferred answer to “how much is on hand?” A product total can be positive even if some individual lot is exhausted; a trace should retain lot-level allocation.

## History and date semantics

`get_transaction_history` joins current transactions to current lines, filters type/date/product/customer, groups the effective lines, and returns corrections. It selectively reads notes, BOL, shipper, customer, order, and adjustment metadata from `effective_record`, but filters and displays the original transaction date/time columns. Its response lines currently omit `product_id`, which weakens frontend lot disambiguation.

`get_daily_production_summary` and `production_day_summary` aggregate make/pack events. The dashboard Production Calendar uses `dashboard_api_production`. These current implementations honor `effective_status`, but they still use the legacy `timestamp` as event time rather than authoritative `business_date` or amended `occurred_at`.

The calendar treats legacy naive timestamps as UTC and converts them to America/New_York before grouping. This matches observed historical storage better than treating them as local, despite an inaccurate source comment. DST and corrections remain risks. The intended Phase 1 date model is explicit `occurred_at` plus plant `business_date`.

## Formula reads and production-warning visibility

`get_batch_formula` (`GET /bom/batches/{batch_id}/formula`) selects the batch's `verification_notes`/`verification_notes_es`, returns its formula, and calls `build_production_warning`. A non-empty English note adds the raw note fields plus the same `production_warning` block used by `/make`. Plain or whitespace-only notes add no warning keys. Both current GPT Action schemas describe this response on the existing `getBatchFormula` operation.

Visibility is intentionally narrow and therefore only partially closes the former “invisible at point of use” gap:

| Surface | Current behavior |
|---|---|
| `/bom/batches/{id}/formula` | Returns raw note(s) and `production_warning` for non-empty batch notes. |
| `/make` preview and commit | Returns `production_warning`; preview appends English note to `preview_message`. |
| `/products/{id}` | Returns the raw product columns, including notes, because it selects `products.*`; not exposed by either current GPT schema. |
| `/products/unverified`, `/products/test-batches` | Can return English `verification_notes` for their special review subsets; not a general production-warning path. |
| `/products/search`, `/products/resolve`, `/bom/products` | Omit verification notes and warning blocks. Search-first alone does not reveal the requirement. |
| Inventory, lot, history, trace, order, and `/dashboard/api/*` reads | Omit production warnings. |
| `/pack` | Resolves products with notes available internally but never calls `build_production_warning`; the notes on finished SKUs 70013–70018 are not surfaced while packing. |

Thus the Floor GPT must proceed from search to formula lookup or `/make` preview to see the requirement. A dashboard-only or direct packing workflow still will not see it.

## Dashboard inventory reads

- `dashboard_api_finished_goods` reads exact configured finished-product names and calculates current effective pounds, cases, and configured panel output.
- `dashboard_api_batches` reads exact configured batch names, effective balances, lots, and batch equivalents.
- `dashboard_api_ingredients` uses configured ingredient categories/names. A product may appear in multiple configured groups.
- `dashboard_api_lot_detail` and `dashboard_api_product_lots` return lot balance and activity drill-down.
- `dashboard_api_search` combines product/lot/customer-ish results for global dashboard search.

Products absent from `dashboard_config.json` are omitted from these primary panel responses. A configured name that no longer matches the product name is reported missing rather than automatically discovered.

## Activity reads

`dashboard_api_shipments` and `dashboard_api_receipts` join current transaction/line views, group lines, and return recent activity. They honor effective void state and effective quantities but use raw header columns/time instead of fully projected header amendments.

The shipping activity is a ledger view, not necessarily a one-to-one view of `shipments`: an order shipment can generate several ledger transactions, and old standalone transactions may lack a shipment header. Receipts similarly show positive receive lines and lot/supplier metadata.

**CONFIRMED FROM CODE:** `dashboard_api_daily_entries` serves public `GET /dashboard/api/activity/daily-entries`. `date` is required. `date_mode=event` filters `ledger_current_transactions.business_date`; `date_mode=entered` filters the America/New_York calendar date of `created_at`; any other mode returns 422. The query joins `ledger_current_transactions` to `ledger_current_transaction_lines`, filters `effective_status='posted'`, and therefore excludes effective voids.

The response envelope contains `date`, `date_mode`, `count`, and `entries`. Each transaction entry includes `transaction_id`, `type`, `event_date`, formatted occurred `date`/`time`, raw and formatted `created_at`, `created_at_source`, `entry_time_reliable`, `late_entry`, `days_late`, `operator_id`, and `lines`. Each line carries `product_name`, `product_id`, `sku`, `lot_code`, and signed `quantity_lb`. A negative shipment/consumption line remains negative; the endpoint does not convert it to an absolute display quantity.

The endpoint is deliberately dashboard-only and absent from both GPT Action schemas, whose operation counts remain 30 and 22.

## Order/readiness reads

`list_sales_orders`, `get_sales_order`, `sales_dashboard`, and dashboard JavaScript feed filters, line detail, readiness flags, dates, quantities, and shipment data. `fulfillment_check` compares each open line independently to the same global product on-hand. It does not reserve stock across competing orders or duplicate same-SKU lines. It also fails to exclude service items in this particular check, so service shortages can be misleading.

The order-matrix export filters services and `no_production` products, groups configured families, and converts pounds to cases. It currently uses full ordered `quantity_lb`, not remaining quantity after shipments, and does not consistently exclude cancelled lines. It can therefore overstate production demand.

## Trace reads

`trace_batch`, `_trace_ingredient_backward`, `trace_ingredient`, and `trace_supplier_lot` reconstruct relationships using lots, raw transaction lines, raw transactions, ILC, customers, and shipment support tables. They can show formula origin, production usage, direct shipments, and supplier-origin information.

Important boundaries:

- the relationship queries largely use raw transaction tables/status and are not correction-aware;
- their final balances may use effective views, so a response can contain corrected on-hand beside uncorrected relationship history;
- ILC itself has no correction projection;
- `fetchone` in creation/origin logic can omit additional events when product+lot is reused;
- `trace_batch` does not recursively follow a packed child finished lot and then its customer shipments;
- supplier-lot trace is immediate usage/direct shipment, not a complete supplier -> batch -> packed FG -> customer recall tree;
- joining both `sales_order_shipments` and `shipment_lines` by transaction can multiply rows when each side has several records;
- frontend traceability fetches only the last 100 history events to build some search indexes, so an older direct movement can be invisible in the browser even if the backend trace response knows it.

The “Complete trace” UI badge means the browser constructed all edges it expected from its limited payload. It is not regulatory proof of one-step-forward/one-step-back completeness.

## Legacy read paths

`dashboard_inventory`, `dashboard_low_stock`, `dashboard_today`, `dashboard_lots`, and `dashboard_production` use legacy views at `/dashboard/*`. Those views aggregate raw rows and some use obsolete enum strings. They should not be mixed with `/dashboard/api/*`, which powers the main static dashboard and is mostly effective-state aware.

## Client-side reconstruction

The browser adds substantial business meaning:

- exact inventory membership and order of products from `dashboard_config.json`;
- family/category recognition from product name substrings;
- 10-lb/25-lb/bagged granola split from `pack_format`;
- batches/pans/cases and pallet conversions;
- calendar layer-1 summary and layer-2 product detail;
- process-line stage classification and “active today” state;
- Sankey node grouping, minimum-flow cutoff, top-N `Other` aggregation;
- trace graph indexing and completeness labels;
- order pallet composition logic;
- all calculations and scheduling in the standalone production board.

When a UI disagrees with an API response, inspect this transformation layer before changing the ledger.

---

# Part 7 — Dashboard Architecture

## Main dashboard shell

`dashboard/index.html`, `dashboard/dashboard.css`, and `dashboard/dashboard.js` implement the main application. There is no component framework; sections, cards, tables, modals, and event handlers are direct DOM code.

Top navigation exposes:

- **Dashboard** — main tabbed operational application;
- **Material Flow** — standalone Sankey page;
- **Production Lines** — conceptual process-flow page;
- **Traceability** — trace graph/search page.

The main application tabs are **Operations**, **Activity**, **Notes**, and **Sales Orders**. It also has a global search overlay, health indicator, theme control, modals/drill-down, and a compact open-order date calendar.

## Operations tab

### Production Calendar card

Business question: “What was actually made or packed on each date?” It calls `/dashboard/api/production` and renders a month grid. The first layer gives family totals; clicking a populated day opens family/product/SKU detail. Its implementation and limitations are in Part 8.

### Finished Goods inventory

Uses `dashboard_config.json` exact groups for 25-lb granola, 10-lb cases (including Graham 31012), retail/bagged products, and coconut finished goods. Cards show on-hand pounds and derived cases; clicking drills to product lots. Business question: “How much sellable finished product is available, and in what lots?”

The six new SS Classic #9 finished products (70013–70018) are not in those exact lists, so this panel currently omits them. This is independent of whether their ledger balances exist.

Case formula is normally `floor(on_hand_lb / case_size_lb)` for panel display. Case sizes come from the database/configured metadata; a missing size prevents meaningful case conversion.

### Batch inventory

Uses a curated list of 18 displayed batch SKUs even though 25 active batch products exist. Shows pounds and batch equivalents using `default_batch_lb`/yield semantics. Clicking gives lot-level detail. Business question: “What work-in-process batch stock is available to pack?” New kosher batches 90025/90026 are not in the configured list.

### Ingredients inventory

Uses configured categories such as grains, nuts/seeds, sweeteners, fruit, chocolate, coconut, packaging, and bulk crumbs. Graham 31011 is in the Bulk Crumbs section. A configured item can appear in more than one category. Business question: “What raw/packaging material is currently available?”

### Pallet indicators

Configured conversion assumptions include 140 cases/pallet for 10-lb cases, 60 for 25-lb, 115 for one retail format, and 144 for another. One configured bagged format is omitted. Inventory cards can display fractional pallets to one decimal. These are display assumptions, not stored pallet inventory.

### Production requirements are not displayed

No main dashboard card, calendar row, lot modal, product search result, or standalone dashboard view renders `verification_notes` or `production_warning`. The 2026-08-12 change reaches formula and make clients but did not change dashboard files. An operator working only from the Dashboard can therefore miss the Kosher Ignition requirement.

## Activity tab

Section order is **Daily Entries, Shipping, Receiving**. Daily Entries first shipped with `dashboard.js?v=27` and `dashboard.css?v=17`; the subsequent reorder/expander deployment advanced the current references to JS v28 and CSS v18. Treat v27+/v17+ as the minimum cache lineage for this feature.

### Daily Entries

Calls public `/dashboard/api/activity/daily-entries` with a selected day and either event-date or entry-date mode. The table renders one row per returned transaction line with Entered time, transaction type, product, SKU, and signed pounds. Reliable entries made on a later plant-local day are highlighted with their lag; migration-backfilled rows are labeled `backfilled` and are not treated as reliable timeliness evidence.

### Shipping

Calls `/dashboard/api/activity/shipments`, renders recent effective posted ship activity with products, quantities, customer/order context, and drill-down. Rows display both Occurred and Entered times, with `backfilled` shown where applicable. The first four transactions are visible by default; “Show all (N more)” reveals the remaining fetched rows and changes to “Show less.” It answers “What left the factory?” It is not a substitute for order/shipment reconciliation because one shipment may span multiple ledger transactions.

### Receiving

Calls `/dashboard/api/activity/receipts`, showing shipper, BOL, product, lot/supplier-lot, cases/pounds, and both Occurred and Entered times; backfilled entry times are labeled. Like Shipping, it shows the first four transactions by default with the same show-all/show-less expander. It answers “What arrived?” Missing supplier/receipt metadata can appear because old/found/imported lots are incomplete.

## Notes tab

Reads public `/dashboard/api/notes`; authenticated create/update/delete/toggle endpoints mutate `notes`. The UI supports note type, due/reminder dates, completion, and related entity metadata. Notes are annotations and task aids, not inventory or production events.

## Sales Orders tab

Displays order rows, filters/status, requested dates, readiness flags, customer/quantity information, order detail/edit modals, line additions/updates/cancellation, state transitions, fulfillment actions, packing slip, and XLSX orders matrix. A compact calendar (`mini-calendar.js`) plots open order requested-ship dates—not production dates.

The UI has two different readiness concepts: calculated fulfillment/readiness from inventory and a manual flag in `sales_order_flags`. Neither reserves inventory.

The pallet utility calculates eligible 10-lb and 25-lb pallet fractions and a mixed/fractional physical pallet count. If a nonzero eligible line has an unknown cases-per-pallet assumption, some outputs deliberately show an em dash rather than false precision.

## Header health, search, refresh, and theme

The health badge calls public `/audit/integrity` and color-codes its coarse score (green at 90+, yellow at 70–89, red below 70); its tooltip lists failing check names, not every defect. Global search calls the dashboard search endpoint and opens matching records. Refresh reloads the current operational datasets and shows a client timestamp. Theme state is a browser presentation preference. None of these controls changes ledger facts.

## Product/lot/search drill-downs

Inventory cards and search results open lot detail sourced by `/dashboard/api/product/{id}/lots` and `/dashboard/api/lot/{lot_code}`. Global search is a navigation/investigation aid. Because lot code uniqueness is per product, callers should pass `product_id` when a code is ambiguous.

## Production Lines (`process-flow.html`)

This standalone view fetches at most the recent 100 make/pack events every 60 seconds and maps product names to four conceptual lines: Granola, Coconut, Bulk, and Pouch. It renders hardcoded stages and interprets transactions as progress/activity. “Active” means a matching transaction occurred today; it is not live machine state. Worker counts (for example 8/10) are placeholders, not database labor records. Yield is output/input from ledger lines; hydration can legitimately show coconut above 100%.

If the API fails, the page renders sample data with a banner. Any user or AI must distinguish that fallback from actual factory evidence. Product-string classification and reuse of a latest case size can distort mixed-product totals.

## Material Flow (`sankey.html`)

This standalone visualization offers 7/30/90/YTD/custom date windows. It fetches up to 500 `make` and `ship` history events, ignores `pack`, drops flows below 50 lb, and groups small/overflow nodes into `Other`. Negative make inputs flow to a string-classified line, positive make outputs flow line-to-product, and ship product flows to customer.

Because pack is omitted, the graph cannot show the essential batch-to-finished conversion and is not a mass-balanced supply-chain model. API failure triggers sample data. The UI's truncation copy still refers to 100 although the request limit is 500.

## Traceability (`traceability.html`)

Provides product/lot, supplier-lot, and shipment-oriented search plus a node/edge graph and details. It calls backend trace endpoints but also builds indexes from only 100 recent history records. It can hide older or direct movements, mis-disambiguate duplicate lot codes because history lines omit product IDs, and show an overly reassuring completeness badge. Treat it as an investigation interface, not a certified recall report.

## Production Board (`scheduler/seven-wells-production-board.html`)

This is a self-contained planning application, not a view of `production_schedule`. It embeds a catalog/routing model, accepts CSV/sample orders, maintains FG/WIP, pins, overtime and settings, runs a greedy 84-day planning algorithm, and persists to `localStorage` key `sw_prodboard_v1`. It neither reads nor writes Factory Ledger APIs.

Its detailed capacity model includes ovens, pan configurations, coconut cycles, pouch/box capacities, labor enumeration, cooling/WIP timing, due-date scoring, Friday modifiers, and forced pins. Those are intended planning assumptions, not facts about actual runs. It may display sample orders/inventory the user entered locally.

---

# Part 8 — Production Calendar

## What it currently displays

The current calendar is a layered actual-production view in the Operations tab:

1. **Layer 1:** a day cell summarizes Made and Packed totals for recognized Granola, Coconut, and Graham families.
2. **Layer 2:** clicking the day opens product/SKU-level detail within each family and activity type.

The August 2026 fixes added Graham visibility, category colors, `pack_format` granola grouping, correct effective-status reads, and product/SKU detail. It no longer intentionally renders placeholder production.

## Backend data chain

`dashboard/dashboard.js`
-> `GET /dashboard/api/production?year=YYYY&month=M`
-> `main.py:dashboard_api_production`
-> `ledger_current_transactions` + `ledger_current_transaction_lines` + `products`
-> positive output lines where type is `make` or `pack` and effective status is posted.

The API returns rows grouped by effective ledger event date, transaction type, product, SKU, and relevant product metadata. It does not return a precomputed family summary. The browser creates the summary.

## Date grouping

The endpoint takes each legacy naive `transactions.timestamp`, treats it as UTC, converts to America/New_York, and groups by the resulting calendar date. Month boundaries follow that converted date. It does not use `transactions.business_date`, and it does not read an amended occurrence date from `effective_record`.

Consequences:

- a Phase 1 corrected business date may not move a calendar entry;
- historical timestamps written under a different convention could shift a day;
- midnight/DST boundaries need explicit tests;
- `created_at` is deliberately not the production date—it is database receipt time.

## Activity selection

Only positive lines on effective posted `make` and `pack` transactions contribute. Receipts, shipments, adjustments, notes, order readiness, schedules, cleaning, case preparation, and support rows are excluded. Negative input lines are used for other analytics, not the calendar's output quantity.

## Family categorization

Browser logic categorizes each product by name heuristics:

- names containing granola-related terms -> Granola;
- coconut-related names -> Coconut;
- Graham cracker crumb names -> Graham;
- anything else -> unclassified/other and omitted from the family summary.

This is not driven by `products.product_category` or a production-family FK. A rename/new family can silently remove actual production from the visible summary; the JavaScript logs a console warning for an unclassified row.

## Unit formulas

### Made granola batches

```text
declared_output_per_batch_lb = default_batch_lb * yield_multiplier
batches_made = round(positive_make_output_lb / declared_output_per_batch_lb)
```

The daily family total is the sum of row-derived batch counts. It is calculated from current product metadata, not stored from the request's `batches` input. If the catalog batch size changes after production, historical displayed count can change.

### Made coconut pans

```text
declared_output_per_pan_lb = default_batch_lb * yield_multiplier
pans_made = round(positive_make_output_lb / declared_output_per_pan_lb)
```

The formula is structurally the same as a batch. “Pan” is a display/business term inferred from family; no pan row exists. The multiplier corrects hydrated coconut display relative to the old default-size-only calculation.

### Packed cases

```text
cases_packed = round(positive_pack_output_lb / case_size_lb)
```

The result is grouped by family/product. Granola is additionally split by `products.pack_format` (`10lb`, `25lb`, `bagged`). Graham 31012 yields `positive pounds / 10`, so the 2026-08-10 4,200-lb pack displays 420 cases.

No rounding in the ledger changes pounds, but `dashboard_api_production` rounds each product/date count to a whole batch/pan/case before the browser sums category totals. This can hide partial units or accumulate product-level rounding error. A missing/zero case size prevents a sound case conversion.

## Stored versus calculated totals

Nothing stores `daily granola batches`, `coconut pans`, or `Graham cases` as a calendar fact. Stored facts are transaction type, output product/lot, pounds, timestamps, and product metadata. All family totals and unit conversions are derived on read.

## SS Classic #9 calendar behavior

No calendar-specific code was added for the kosher family. The existing name heuristic sees “Granola” in 90025/90026 and 70013–70018, so future posted `make`/`pack` output will fall under the Granola family. Made counts use the new 323-/348-lb batch denominators. Packed 70014/70017 and 70015/70018 use their `25lb`/`10lb` formats and case sizes. Bulk-per-pound 70013/70016 have null `pack_format` and case size, so their pack output is omitted from classified packed-case totals. The calendar response/display does not carry the Kosher Ignition warning.

## Exclusions and blind spots

- actual make/pack for an unrecognized family is omitted from layer 1;
- configured `pack_format` is sparse; an unmapped granola pack can fail to enter the intended format bucket;
- four active finished products lack case size, including bulk-per-pound 70013/70016;
- `no_production` is not consulted by the calendar; if a product marked resale/no-production receives a pack transaction, the name heuristic may still show it;
- voids through effective status are excluded correctly, but amended dates are not;
- box/label preparation, labor, scrap, downtime, and sanitation cannot appear because there is no ledger event;
- schedule rows never appear because the calendar is actual-only.

## Graham root cause and current status

The backend already returned 31012 pack output. Before the 2026-08-11 fix, frontend family/render logic recognized only Granola and Coconut, so Graham rows were classified as `other` and never rendered. Commits `078575f` and `1359df5` added Graham classification/display; `de39c54` records deployment. **Current implementation can retrieve and show Graham packing.** The architectural fragility remains because recognition is a string rule rather than stable data.

## Supporting a durable snapshot-to-detail design

The basic two-layer interaction now exists. To make it architecture-grade rather than heuristic:

1. add stable production-family and display-unit metadata (or normalized tables) to the product/catalog model;
2. version effective conversion metadata, or persist event-time batch/case denominator on the transaction/line;
3. use authoritative effective `business_date`/`occurred_at` in queries;
4. return a typed backend summary with an explicit `unclassified` bucket, while retaining raw product rows for drill-down;
5. define whether coconut “pan” equals an API-request batch, an output-weight denominator, or a physical pan count;
6. define additional families/categories and ensure new products cannot silently vanish;
7. if packaging stages/labor are desired, model and capture them as actual events instead of manufacturing them from names in the UI.

---

# Part 9 — Business Logic and Calculations

This section records formulas as implemented. A “unit” displayed by the UI may be a derived convention, not an independently counted physical object.

## Effective inventory

**Formula:**

```text
lot_on_hand_lb = SUM(ledger_current_transaction_lines.quantity_lb)
                 for the lot
                 where joined ledger_current_transactions.effective_status = 'posted'

product_on_hand_lb = SUM(lot_on_hand_lb for all product lots)
```

**Inputs:** effective posted signed lines. **Output:** pounds on hand. **Source:** `main.py` constant `POSTED_LINES`, `lot_on_hand`, `_inventory_detail_for_products`, dashboard inventory queries. **Assumptions:** every physical movement has exactly one correct line; void/correction projection is complete; no unrecorded physical movement. `BALANCE_EPSILON` tolerates small numeric noise in deduction checks.

## Receipt quantity

**Formula:** `received_lb = cases * case_size_lb`.

**Inputs:** operator case count and pounds/case. **Output:** positive receive line and header metadata. **Source:** `main.py:receive`. **Assumptions:** every case has the given weight; the operator uses the correct SKU and supplier/system lot.

## Make output and inputs

**Formula:**

```text
formula_weight_lb = default_batch_lb * requested_batches
make_output_lb    = formula_weight_lb * yield_multiplier
component_needed_lb = batch_formulas.quantity_lb * requested_batches
```

**Inputs:** current product batch size/yield, requested batches, current formula rows. **Output:** one positive batch line plus negative component lines. **Source:** `main.py:make`. **Assumptions:** `default_batch_lb` represents base formula weight; multiplier represents finished yield; formula quantities are independently correct. The code does not reconcile the output formula with component sum.

For an excluded component, `would_need_lb` is calculated but actual inventory consumption is zero. This is how water is represented in hydrated coconut.

## Production-warning rule

**Rule:** `trim(products.verification_notes)` non-empty on the product being made -> return `production_warning` on formula lookup and make preview/commit. Copy `verification_notes` verbatim; copy optional non-empty `verification_notes_es`; construct a relay message; append English text to make preview. **Source:** `main.py:build_production_warning`, `make`, `get_batch_formula`. **Assumptions:** the product note is current, relevant to every run, and the client follows the relay/confirmation instruction. **Non-enforcement:** no database constraint, run-level acknowledgment, operator identity, or ignition/compliance event is created.

## Pack conversion

**Formula:**

```text
pack_output_lb = cases * (request.case_weight_lb or target.case_size_lb)
base_source_consumption_lb = pack_output_lb
```

**Inputs:** cases, case size, source and target products/lots. **Output:** negative source pounds and equal positive target pounds. **Source:** `main.py:pack`. **Assumptions:** lossless 1:1 pound transfer, except any separately consumed add-ins. Packaging BOM is not part of the formula.

## Pack add-ins

When the target's parent batch formula contains the selected source plus other non-excluded ingredients:

```text
ratio = pack_output_lb / formula_base_source_quantity_lb
add_in_needed_lb = round(formula_add_in_quantity_lb * ratio, 2)
```

**Inputs:** target parent formula, selected source, output pounds. **Output:** extra negative lines and ILC rows. **Source:** `main.py:resolve_pack_add_ins`, `pack`. **Assumptions:** the formula ratio is valid at the packing hopper and a two-decimal ingredient deduction is acceptable.

## FIFO lot allocation

Eligible lots have a positive effective balance. They are ordered by `COALESCE(received_at, created_at) ASC`; the algorithm takes `min(remaining_required, lot_available)` until fulfilled. Explicit lot allocations must sum to required pounds within 0.01 lb. **Source:** `make`, `pack`, `ship`, and `ship_order`. Missing receipt dates fall back to lot creation time, which may not reflect physical receipt chronology.

## Adjustments

**Formula:** `new_lot_balance = current_effective_balance + requested_delta_lb`.

**Inputs:** signed delta. **Output:** one adjust transaction/line. **Source:** `main.py:adjust`. **Assumptions:** the count/reason is authoritative. A special negative adjustment can intentionally bypass normal no-negative controls.

## Case and batch equivalents in inventory

Typical UI/API conversions are:

```text
case_equivalent = on_hand_lb / case_size_lb
displayed whole available cases = floor(on_hand_lb / case_size_lb)  [main inventory panel]
batch_equivalent = on_hand_lb / declared batch unit size
```

Different endpoints/cards use floor, round, or fractional formatting. An AI must cite the exact view when comparing “cases” and never assume every response uses the same rounding.

## Production Calendar

Per product/date, backend SQL sums positive output pounds and then computes:

```text
made_unit_size_lb = default_batch_lb * yield_multiplier
displayed made units = round(total_positive_make_lb / made_unit_size_lb)
displayed packed cases = round(total_positive_pack_lb / case_size_lb)
```

The browser sums those already-rounded product values into family/format groups. **Source:** `main.py:dashboard_api_production`; `dashboard/dashboard.js:productionBatchCount`, `productionUnitCount`, `buildProductionDaySummary`. **Assumptions:** current metadata applies historically; product names encode family; `pack_format` is complete for granola.

## Pallets

For the main order pallet utility:

```text
10-lb case: 140 cases/pallet
25-lb case: 60 cases/pallet
line pallet fraction = cases / cases_per_pallet
physical pallets for a line = ceil(line pallet fraction)
order pallet fraction = SUM(eligible line pallet fractions)
mixed physical pallets = ceil(order pallet fraction)
```

**Source:** `dashboard/pallet-calculations.js`. **Assumptions:** only 10/25-lb case formats are recognized by metadata/text. Services/non-weight lines are excluded. If any positive eligible line is unmapped, the aggregate display is `—`.

The inventory panel additionally has format-specific configured values (including 115 and 144 cases/pallet for selected retail formats) in `dashboard_config.json`, so inventory and order pallets do not share one universal table.

## Sales-order normalization and remaining demand

For physical lines, cases/bags/boxes are converted using case/product metadata to `quantity_lb`; services may be zero-weight. Remaining demand used by the API scheduler is:

```text
remaining_line_lb = quantity_lb - quantity_shipped_lb
net_product_need_lb = SUM(remaining lines) - current finished-product on_hand_lb
```

**Source:** `main.py:create_sales_order`, `_load_demand`, `_simulated_allocation`. **Assumptions:** `quantity_shipped_lb` is synchronized with effective shipment state—currently false after a ship void.

## Production requirements and batch rounding

The production requirements/scheduler use:

```text
batches_required = ceil(net_batch_need_lb / default_batch_lb)
planned_output_lb = batches_required * default_batch_lb
overproduction_lb = planned_output_lb - net_batch_need_lb
ingredient_required_lb = formula_quantity_lb * batches_required
ingredient_shortage_lb = max(0, required_lb - on_hand_lb)
```

**Source:** `production_requirements`, `_simulated_allocation`, `_explode_ingredients`. **Assumptions/defects:** ignores `yield_multiplier`; ignores the `product_bom.quantity` of the batch component; only accepts a finished BOM component whose type is `batch`; does not create a separate pack requirement; ignores `no_production`.

## API scheduler capacity

Weekends are skipped. Friday capacity is multiplied by a configurable modifier. When a line is first activated:

```text
available_labor = total_workers - labor_already_used_that_day
choose first capacity mode that fits labor
day_batch_capacity = max(1, int(mode.batches_per_day * day_modifier))
scheduled_batches = min(remaining_batches, day_batch_capacity)
```

**Source:** `_build_schedule_calendar`, `_schedule_runs_to_days`. Modes are ordered default first, then fewest workers; this is not throughput optimization. `pallets_per_day` and `bags_per_day` are loaded but ignored by the scheduling loop. Therefore Bulk Packing/Pouch modes with null `batches_per_day` receive `max(1, 0) = 1` nominal batch/day if a requirement reaches them.

## Process-flow yield

The conceptual Production Lines page derives event yield as `positive output pounds / absolute negative input pounds * 100`. Hydrated coconut can exceed 100%. This is descriptive arithmetic on ledger lines, not a quality-spec yield calculation, and formula exclusions further affect interpretation. **Source:** `dashboard/process-flow.html`.

## Integrity score

```text
score starts at 100
for each failing check (not each failing row):
  critical -> -10
  major    -> -5
  minor    -> -1
info does not subtract
minimum score = 0
```

**Source:** `main.py:audit_integrity`. A check with 590 defects costs the same as a check with one defect. The score is a triage indicator, not a compliance metric.

---

# Part 10 — Terminology Dictionary

| Term | Meaning in this system |
|---|---|
| Actual / actual production | An effective posted `make` or `pack` transaction with a positive output line. |
| Adjustment | A signed inventory delta recorded as transaction type `adjust`; includes counts, found stock, and exceptional reconciliation. |
| Batch | Ambiguous: a catalog `product.type='batch'`; one requested `/make` unit; or a derived calendar count from output pounds. It is not a run table. |
| Batch formula | `batch_formulas`: pounds of each component per nominal batch, used by `/make`. |
| Baked / made / produced | Calendar “Made” normally means a `make` output. It proves ledger production, not a recorded oven stage. |
| BOM | Ambiguous: `batch_formulas` is the executable make recipe; `product_bom` maps finished to components for planning/admin; `boms`/`bom_lines` is unused legacy structure. |
| Business date | Plant-local operational date on `transactions.business_date`; intended for factual cutoff/grouping but not yet used consistently. |
| Case | Derived finished-product unit, normally pounds divided by `case_size_lb`; pack requests also store case count metadata. No individual case record exists. |
| Certification | Append-only assertion that a business date is complete as of `certified_at`; live deployment contains only smoke data. |
| Commingled | Multiple supplier lots represented within one system lot/receipt support mapping. Exact physical separation must be captured by separate receipts/lots. |
| Correction | Append-only amend/void/restore event projected over an original transaction or line. |
| Created time | Database receipt timestamp (`created_at`), not necessarily when factory activity occurred. |
| Current/effective ledger | The original ledger after applying the latest correction, exposed by `ledger_current_*` views. |
| Customer label / private label | A distinct finished SKU/metadata convention. Label application and consumption are not separately recorded. |
| Dashboard | Static presentation client and its read APIs; not the source of inventory truth. |
| Effective status | Posted/voided state derived by `ledger_current_transactions`, including correction-based void/restore. |
| Finished good | `products.type='finished'`, normally inventory ready to ship; may also be resale, copack, service-adjacent, or `no_production`. |
| Formula exclusion | Component still present in recipe math but not deducted from inventory, commonly water. |
| Ingredient | `products.type='ingredient'`; can be received, used by make/pack, sold, or marked no-production. |
| ILC | `ingredient_lot_consumption`; positive record of which lot a make/pack consumed, duplicating negative ledger movement for trace. |
| Kosher Ignition | Product-specific production requirement on SS Classic #9 SKUs: the owner or designated messenger must turn on the oven before production. Stored as `verification_notes`; no compliance event is recorded. |
| Label | Product metadata/name concept. There is no label entity, label lot, or label-application event. |
| Ledger | The durable transaction headers, signed lines, lots, and effective corrections—not every support/planning table. |
| Lot | Product-specific identity for traceable inventory. Text code is unique only within a product. |
| Make | Formula-driven conversion that adds a batch product and removes ingredient lots. |
| Mix | Business synonym sometimes embedded in batch/planner language; no separate mix entity. Needs owner definition versus batch/pan. |
| No production | `products.no_production=true`; intended resale/pass-through flag. Not enforced by all planning/calendar/write code. |
| Odoo code / SKU | `products.odoo_code`, the main business catalog code; no live Odoo sync exists. |
| Occurred time | Time factory activity occurred (`occurred_at`); more semantically correct than `created_at`, but reads still often use legacy `timestamp`. |
| Operator | In current audit fields, usually `legacy-shared-key`; not a verified person. |
| Pack | Lossless pound conversion from source material to target product, plus optional formula add-ins. Not equivalent to make. |
| Packed | Positive output on transaction type `pack`; cases are derived from pounds/case size. |
| Packaging | `products.type='packaging'`; merely listing it on `product_bom` does not make `/pack` consume it. |
| Pallet | Display/planning conversion from cases. No durable physical pallet/license-plate entity. |
| Pan | Calendar/planner coconut unit. In the calendar it equals rounded output pounds divided by declared output per batch; not a stored pan. |
| Posted | Effective transaction participates in inventory. Raw status and effective status can differ. |
| Production warning | Advisory API block generated from a non-empty batch `verification_notes`; currently returned by formula lookup and `/make`, not a backend-enforced prerequisite. |
| Production event / run | No dedicated table. In factual usage, one `make` or `pack` transaction is the closest event/run header. |
| Production date | Calendar date derived from legacy event timestamp; intended canonical term should be `business_date`. |
| Production line | Capacity/planning entity (`production_lines`), not written on make/pack transactions. Actual output cannot be definitively attributed to a line. |
| Production schedule | Planned row in `production_schedule` or a separate local browser plan. Never assume it happened. |
| Receive | Positive inbound inventory transaction tied to lot, shipper/BOL, and supplier lot metadata. |
| Resale/pass-through | Product marked `no_production`; should generally be fulfilled from receipts, not planned for internal production. |
| Shipment | Ambiguous: ledger `ship` transaction; `shipments` header; `shipment_lines`; or `sales_order_shipments`. Inventory truth is the effective negative ledger line. |
| Ship | Negative inventory transaction to a customer/order. |
| Source of truth | The most authoritative layer for a question; for quantity movement, effective posted ledger lines. |
| Supplier lot | Vendor/manufacturer lot code on `lots`; distinct from CNS system lot. |
| System lot | CNS lot code used to identify a product-specific inventory pool. |
| Void | Effective removal of a transaction through a correction (new model) or raw status (legacy). No reversal line is required. |
| Verification notes | Mutable English/optional-Spanish product metadata used for review context and, on batch formula/make paths, production warnings. Not historical run evidence. |
| WIP | Usually a batch-product balance available to pack. The standalone planner also has local WIP timing; it is not database actual. |
| Yield multiplier | Product factor multiplying nominal default batch pounds to calculate make output and calendar unit denominator. |

---

# Part 11 — Source-of-Truth Rules

## Authority hierarchy

1. **Live production Postgres + deployed effective schema** for current records, constraints, and balances.
2. **Current `main` implementation at the deployed/matching commit** for how API requests read/write those records.
3. **Current exact configuration and Action contracts** for what the dashboard/GPT can expose.
4. **Current tests** for protected regression behavior—not proof of uncovered behavior.
5. **Deployment changelogs/docs** as historical evidence that a change was shipped.
6. **Old context/status/audit documents** only after verifying against current code/live state.
7. **Standalone planner/sample/fallback data** as hypothetical or local user input, never an operational fact.

`CONTEXT.md` describes an older 2.5-era system; `STATUS.md` was classified on 2026-08-05 before Phase 1 and migration 040; the March trace audit predates several fixes. They explain evolution but must not override the current code.

## Question-to-source matrix

| Question | Authoritative source | Derived/supporting source | Never sufficient alone |
|---|---|---|---|
| How many pounds are on hand? | Sum of effective posted lines | inventory API/dashboard | raw legacy view, planner inventory |
| What lot holds them? | effective lines grouped by `lots` | lot API | product total alone |
| Was production performed? | effective posted make/pack positive line | calendar/history | schedule row, process-flow sample |
| How many calendar batches/pans/cases? | output pounds + event-time-intended conversion metadata | current calendar calculation | stored request `cases` alone, planner |
| What was consumed? | effective negative lines; corroborate with ILC | trace endpoint | formula alone |
| Which supplier lot? | `lots.supplier_lot_code` and receipt lineage | receipt transaction/BOL | system lot naming guess |
| What shipped from inventory? | effective negative ship lines | shipment support tables | `quantity_shipped_lb` alone after void |
| What remains on an order? | order line state reconciled with effective shipment facts | sales APIs | fulfillment check as reservation |
| What is planned? | `production_schedule` for API-confirmed plan; browser state for that local plan | scheduler response | ledger actuals |
| Did plan occur? | effective actual make/pack and an explicit reconciliation (currently absent) | human comparison | schedule status alone |
| Is a day final? | current certification plus late-record check | transaction history | date has passed |
| What production restriction applies? | current `products.verification_notes` / `_es` | formula or `/make` `production_warning`; GPT relay rule | dashboard, search result, or a past make transaction |

## Example: “18 granola batches on August 5”

The defensible chain is:

1. retrieve all effective posted `make` output lines whose authoritative business date is August 5 (the current endpoint instead uses converted legacy timestamp);
2. confirm each output product belongs to the Granola production family;
3. divide each product's output pounds by its event-applicable `default_batch_lb * yield_multiplier` and apply the documented rounding policy;
4. sum the product counts;
5. retain the transaction and line IDs as drill-down evidence.

The calendar's visible “18” is a derived claim. Today, product family and denominator are current mutable metadata, so exact historical reproduction is weaker than it should be.

## Actual versus derived versus manually maintained

- **Canonical actual:** effective transaction/line/lot quantity facts and immutable correction events.
- **Derived:** inventory totals, case/pallet/batch/pan counts, dashboard family summaries, readiness, flow graphs, integrity score.
- **Mutable operational master data:** products, case/batch/yield attributes, formulas, product BOM, customers, orders, line assignments/capacities.
- **Manual annotations:** notes, readiness flags, reasons.
- **Plan:** `production_schedule` and standalone board.
- **Legacy/noncanonical:** raw aggregation views, superseded schemas/prompts, sample fallback data.

---

# Part 12 — Schedule Versus Actual Production

## The strict boundary

```text
WHAT WE PLANNED TO MAKE
  API scheduler suggestion/confirmation -> production_schedule
  OR standalone board -> browser localStorage

WHAT THE FACTORY ACTUALLY MADE
  effective posted make/pack transaction -> positive output line
```

There is no automatic bridge. Schedule confirmation creates no inventory. Make/pack creates no schedule completion. No variance table links planned quantity, actual quantity, product, line, or date.

## Backend/API scheduler

Entry: `POST /schedule` with actions `suggest`, `confirm`, or `current`; helpers `_load_demand`, `_load_finished_inventory`, `_load_bom_structure`, `_simulated_allocation`, `_explode_ingredients`, `_schedule_runs_to_days`.

Demand includes order headers in confirmed/in-production/ready, line statuses pending/partial, and requested ship date null or within horizon. It subtracts `quantity_shipped_lb`, consolidates by product, allocates current finished stock, then attempts to turn remaining finished demand into a batch component using `product_bom`. Existing batch inventory is allocated before new batches.

It produces earliest and latest scenarios, ingredient-risk summaries, overproduction from whole-batch rounding, warnings, and unscheduled demand. Confirm upserts rows by date/line/product into `production_schedule` with linked order numbers and assigned workers.

### Live line capacities

| Line | Mode | Workers | Stored capacity |
|---|---|---:|---|
| Granola Baking | default 2-worker | 2 | 9 batches/day |
| Granola Baking | 3-worker | 3 | 16 batches/day |
| Coconut Sweetened | standard | 2 | 12 batches/day |
| Bulk Packing | default 25-lb | 2 | 4 pallets/day |
| Bulk Packing | 10-lb | 2 | 9 pallets/day |
| Pouch | standard | 3 | 7,500 bags/day |

At audit, `production_schedule` had zero rows. Thus there was no persisted live plan to reconcile.

### Backend scheduler limitations

- ignores `products.no_production`, so resale products can enter demand;
- requires a finished product's `product_bom` to include a component with `type='batch'`; Graham 31012 points to ingredient 31011 and cannot schedule;
- cannot schedule the new SS Classic #9 family from current planning data: 70013–70018 have only `parent_batch_product_id` (no `product_bom`), and none of 90025/90026/70013–70018 has a `product_line_assignment`;
- ignores `product_bom.quantity` when converting finished pounds to batch need;
- ignores `yield_multiplier` in batch output;
- creates batch-make runs but no separate finished pack runs;
- explodes only the selected batch's direct formula and does not recursively plan nested batch BOMs;
- loads but does not use `pallets_per_day` or `bags_per_day`;
- picks the default/first labor-fitting mode, not the maximum-throughput mode;
- hardcodes output line ordering/codes;
- has no actual completion, line downtime, shift, changeover, or material reservation.

`production_requirements` is a related one-off explosion endpoint with similar batch-component/yield/BOM limitations. It rejects Graham 31012 as a producible finished good because its relevant component is an ingredient.

## Standalone production board

`dashboard/scheduler/seven-wells-production-board.html` contains a separate catalog and planning engine. It persists only to `localStorage` and can import CSVs or load sample orders. It embeds about 58 SKUs despite a nearby comment claiming 57.

Default assumptions include 10 workers, 495 minutes/day, 75% Friday, 50% same-day WIP, zero default changeover, a three-oven granola model with pan/batch choices, coconut labor/cycle constraints, pouch 8,500 bags/day, 10-lb box 1,400 cases/day, 25-lb box 240 cases/day, and Graham/repack 250 cases/day as a placeholder. Some values differ from database capacities (for example database pouch is 7,500 bags/day).

The greedy planner scans up to 84 days, enumerates labor allocations, scores due-date urgency, consumes local FG first, tracks local WIP/cooling, handles coconut next-workday constraints, and can force pinned work while flagging infeasibility. It intentionally does not leave capacity idle simply because demand is not yet due.

Graham 31012 is routed as `repack` at the placeholder capacity; 31011 is `packonly`, described as sold-as-purchased/unconstrained with routing to verify. That is a planning hypothesis, not the actual API model.

## Required design for plan-versus-actual

A reliable reconciliation would need stable schedule-run IDs, an actual-event link on make/pack, planned and actual units with event-time conversion metadata, line/work-center attribution, status rules driven or confirmed by actuals, variance/reason fields, and explicit handling for make and pack as separate operations. Until then, an AI must discuss them independently.

---

# Part 13 — Data Quality and Integrity

## Implemented safeguards

- database PK/FK/unique constraints for core identity and relationships;
- product type and selected status/format check constraints;
- transactional writes with rollback and lot row locks;
- effective on-hand validation and FIFO allocation for ordinary deductions;
- explicit preview/commit contract in GPT instructions;
- structured write receipts and error envelopes;
- tiered product resolution/disambiguation rules;
- shared-key authorization on operational/admin routes;
- append-only original transactions/lines and append-only correction/certification events after Phase 1;
- database-forced `created_at` and `created_at_source` on audited tables;
- effective current views for void/restore/line amendments;
- private-label merge-keyword protection;
- integrity endpoint checks and database-backed regression tests;
- service/no-production filters in selected order/export/dashboard paths;
- exact guarded `pack_format` backfill script.
- formula and make warning responses for non-empty product verification notes, with a Floor GPT relay-before-commit rule.

## Live point-in-time integrity result

At 2026-08-12 12:34 America/New_York, `/audit/integrity` still returned **85/100**:

| Check | Result | Detail |
|---|---|---|
| Negative effective lot balances | Pass | 0 |
| Posted make missing ILC | Pass | 0 detected |
| Ship missing shipment line after cutoff | Fail | transaction 531 |
| Lots missing `received_at` | Fail | 590 of 962 lots |
| Received lots missing supplier-lot code | Fail | 49 of 176 receipt-entry lots |
| Packed FG missing case size | Pass | 0 among products with positive posted pack history |
| Floating-point dust | Pass | 0 |
| Raw voided transaction count | Info | 33 |

The effective ledger had 34 voided transactions, demonstrating that the raw void count misses the one correction-based void. Passing checks are only as good as their queries.

## How inaccuracies can enter

### Capture omissions

Physical production/receipt/shipment may never be entered; label/box/labor/scrap/cleaning data cannot be entered structurally; old supplier/receipt metadata is incomplete. No sensor/Odoo integration independently verifies operator capture.

### Duplicate commits

There is no general idempotency key. A retried commit after an ambiguous network response can duplicate production or receipt activity unless the client uses the receipt and checks history. Unique lot rules do not prevent duplicate transactions into the same lot.

### Wrong product or lot

Names, approximate resolution, auto-created customers, and same lot text across products create ambiguity. GPT search-first rules reduce but do not eliminate it. Reassignment/merge correction paths only partially synchronize ILC.

### Time/date errors

Legacy naive timestamp, new occurred/business date, immutable creation time, timezone conversion, and correction JSON coexist. Calendar/history ignore some supported amendments, so a corrected date can remain visually wrong.

**CONFIRMED FROM CODE/LIVE DEPLOYMENT RECORD:** migration 039 gave the live pre-2026-08-11 ledger population a single backfill `created_at` equal to the migration run time, approximately 2026-08-11 10:32 AM America/New_York, with `created_at_source='migration_backfill_039'`. That timestamp is not the historical entry time. Timeliness logic must gate on `created_at_source='database'`; Daily Entries reports backfilled rows as `entry_time_reliable=false`, labels them `backfilled`, sets `days_late` to null, and never flags them late.

### Unit/catalog drift

Case size, batch size, yield, formula, product BOM, and pack format are mutable and unversioned. Historic batches/cases can be recalculated using today's denominators. Formula mass need not match output; packaging BOM is unenforced.

### Void/order divergence

A ship void restores effective inventory but not order shipped quantities/status or shipment support rows. Trace/history/order views can disagree.

### Raw/effective query mixing

Legacy views, trace relationships, and portions of integrity checks query raw status/lines, while inventory uses effective lines. A correction can cause two valid-looking screens to differ.

### UI exclusions

Exact configured names omit nonlisted products. Family heuristics and nullable `pack_format` omit actual calendar rows. Sankey omits pack; process/trace pages cap history and can show fallback samples.

### Planning confusion

Standalone planner and API schedule assumptions can be mistaken for actual activity. Neither reserves inventory nor reconciles actuals.

## Missing safeguards

- individual identity, roles, secret-free client delivery, and auditable operator authentication;
- idempotency keys/request fingerprints for every commit;
- event-time snapshot/version of product/formula/conversion metadata;
- formula/output mass tolerance and required-formula/type validation on `/make`;
- enforced source-target/BOM/packaging rules and scrap accounting on `/pack`;
- canonical effective header projection in every read;
- correction-aware ILC/trace and recursive forward/back recall graph;
- atomic shipment-void reconciliation across ledger, order, and shipment tables;
- supplier lot and received time completeness gates appropriate to lot source/type;
- stable production taxonomy and an explicit unclassified alert;
- production event fields for line, shift, worker, stage, downtime, and schedule link;
- reservations/allocation across competing orders;
- operational daily certifications and late-entry workflow;
- meaningful volume-weighted/data-severity integrity metrics.
- server-side acknowledgment/enforcement for production restrictions, warning propagation to search/pack/dashboard surfaces, and a run-level record of who satisfied the requirement.

## Test coverage boundary

Current pytest collection is 79 cases (72 explicit Python test functions plus parametrization), and the separate JavaScript pallet suite has four cases. `tests/test_production_warning.py` adds six database-backed tests: warning/no-warning formula responses, warning on make preview and commit, no warning for ordinary products, and whitespace-only suppression. `tests/test_daily_entries.py` adds three database-backed tests covering same-day entry metadata and signed line quantities, late-entry plus entered-date filtering, and effective-void exclusion. The shipped post-merge run passed 79/79 against the refreshed test database. Coverage is strongest around Phase 1 append-only/corrections, void balances, response envelopes, notes auth, service lines, order matrix, customer resolution, calendar calculations, production-warning response behavior, Daily Entries timeliness behavior, and read-only retry.

`tests/schema/schema.sql` now contains the Phase 1 append-only triggers/current views/certifications and the migration 040 pack-format structure. A local test database rebuilt from this snapshot therefore exercises those structures; it still contains no production catalog data unless a test seeds it.

There is little/no direct regression coverage for full normal receive/make/pack/standalone-ship behavior, recursive trace completeness, backend scheduler semantics, integrity-query correctness, or end-to-end browser pages. Passing tests must not be generalized to those domains.

---

# Part 14 — Known Gaps, Inconsistencies, and Technical Debt

Each issue below distinguishes the current problem from a possible direction. Directions are architectural options, not approved requirements.

## 1. Shared credential and no accountable operator identity

**Problem:** one API key protects all users and is embedded in public/static client and GPT configuration; CORS allows every origin. Audit rows resolve to `legacy-shared-key`. **Why it matters:** possession of the client/config can grant write/admin access, and records cannot be attributed to a person. **Where:** `main.py:verify_api_key`, `_operator_id`, CORS setup; dashboard JS; generated GPT instructions. **Likely consequence:** unauthorized changes and weak audit/nonrepudiation. **Possible direction:** individual identity, roles/scopes, short-lived server-issued credentials, secret removal from clients/query URLs, and explicit actor propagation.

## 2. Header corrections are recorded but not actually projected

**Problem:** `ledger_current_transactions.effective_record` contains amended date/time/metadata, while most queries read original columns. **Why it matters:** a successful correction can leave history/calendar/activity visibly unchanged. **Where:** migration 039 current view; `get_transaction_history`, `dashboard_api_production`, shipment/receipt APIs. **Likely consequence:** operators distrust corrections and dates disagree. **Possible direction:** expose typed effective columns in the view and require every read to use them; regression-test each supported field.

## 3. Raw and effective ledger paths coexist

**Problem:** canonical inventory uses current views, but legacy views, trace joins, and integrity checks still read raw rows/status. **Why it matters:** the same transaction can be voided in one screen and posted in another. **Where:** legacy schema views and `/dashboard/*`; trace functions; `audit_integrity`. **Likely consequence:** conflicting totals and recall paths. **Possible direction:** one reusable effective ledger SQL/view layer with deprecation/removal of raw aggregation endpoints.

## 4. ILC is outside the correction model

**Problem:** ILC duplicates consumption but has no current/effective projection. Merge mutates it; reassignment leaves it stale. **Why it matters:** trace origin can disagree with corrected negative ledger lines. **Where:** `ingredient_lot_consumption`, `reassign_lot`, `merge_lots`, trace functions. **Likely consequence:** incorrect product/lot lineage after data repair. **Possible direction:** derive consumption from effective negative lines or make ILC append-only/correction-aware with invariants between the two representations.

## 5. Shipment void does not reopen the order

**Problem:** correction-based void restores inventory but leaves shipment rows, shipped quantities, line status, and order status unchanged. **Why it matters:** inventory may be back while the order remains shipped. **Where:** `void_transaction` versus `ship_order` side effects. **Likely consequence:** fulfillment, invoice, and customer history errors. **Possible direction:** domain-specific atomic shipment void/restore orchestration or effective shipment/order projections derived from ledger state.

## 6. Traceability is not a recursive correction-aware recall graph

**Problem:** trace endpoints follow selected immediate edges, often raw; batch trace does not traverse pack children to customers; frontend indexes only recent history. **Why it matters:** “complete” can omit downstream exposure. **Where:** `trace_batch`, `trace_ingredient`, `trace_supplier_lot`, `traceability.html`. **Likely consequence:** false assurance and manual recall work. **Possible direction:** canonical graph edges from effective lines/ILC/support tables, recursive bounded traversal in both directions, deduplication, explicit incompleteness reasons, and recall tests.

## 7. Product family is inferred from mutable names

**Problem:** Granola/Coconut/Graham classification lives in JavaScript string rules and exact configuration lists. **Why it matters:** rename/new product can silently vanish from calendar/panels. **Where:** `dashboard.js:getProductCategory`, `dashboard_config.json`, process/Sankey classifiers. **Likely consequence:** valid production is underreported. **Possible direction:** normalized production family/routing/display-unit attributes with validation and a mandatory unclassified queue.

## 8. Product structures are duplicated and unenforced

**Problem:** `batch_formulas`, `product_bom`, `parent_batch_product_id`, and empty legacy `boms`/`bom_lines` overlap. `/pack` does not enforce `product_bom`. **Why it matters:** planning, source resolution, and execution can disagree. **Where:** schema, `/make`, `/pack`, requirements/scheduler, admin routes. **Likely consequence:** impossible schedules, wrong conversions, unused packaging BOM. **Possible direction:** define one versioned structure per purpose—execution recipe, pack/routing BOM, packaging BOM—and enforce explicit relationships.

## 9. Catalog changes rewrite historical interpretation

**Problem:** batch size, yield, case size, formula, and category are mutable without event-time snapshots. **Why it matters:** a historic 350-lb output can change from one batch to another count after a catalog edit. **Where:** `products`, formula/BOM admin, calendar/inventory conversions. **Likely consequence:** nonreproducible reports. **Possible direction:** versioned catalog/recipes with effective dates or immutable transaction snapshots of denominators and recipe version.

## 10. Make permits mass-inconsistent or invalid production

**Problem:** output is independent of formula sum; product type/formula presence/mass tolerance are not enforced. **Why it matters:** ledger can manufacture/lose large quantities. **Where:** `main.py:make`; live SKU 90008 evidence. **Likely consequence:** inflated WIP and impossible yield. **Possible direction:** required batch type/formula, preview mass reconciliation, explicit yield/loss/water model, guarded override with accountable approval.

## 11. Pack omits packaging, scrap, and physical stages

**Problem:** 1:1 source pounds become output; labels/bags/boxes from `product_bom` are not consumed; no scrap/case erection stage. **Why it matters:** finished quantity does not prove packaging trace or mass reconciliation. **Where:** `main.py:pack`; Graham BOM/workflow. **Likely consequence:** inaccurate packaging inventory and incomplete recall/operations data. **Possible direction:** pack-run entity or richer transaction subtype with packaging lot consumption, expected/actual output, scrap, stages, and label identity.

## 12. No labor, shift, line, machine, or downtime actuals

**Problem:** actual make/pack transactions do not identify production line or workforce. **Why it matters:** management cannot compute factual capacity/utilization/downtime from the ledger. **Where:** transaction schema and write requests; process-flow page substitutes conceptual/hardcoded values. **Likely consequence:** planning assumptions masquerade as performance data. **Possible direction:** event/run fields or related operational-event tables with controlled capture burden.

## 13. Two unrelated schedulers

**Problem:** API/database scheduler and browser-local production board have different data, capacities, and algorithms. **Why it matters:** two users can discuss “the schedule” while viewing incompatible plans. **Where:** `main.py:/schedule`; standalone scheduler HTML/localStorage. **Likely consequence:** duplicated maintenance and no shared plan. **Possible direction:** choose a canonical planning domain/service; persist scenarios/version; make all UIs its clients.

## 14. Scheduler math does not model packing capacities

**Problem:** the API loop only uses `batches_per_day`; pallet/bag capacities load but are ignored, and finished demand becomes only batch-make runs. **Why it matters:** Bulk/Pouch output is nonsensically treated as one “batch” per day. **Where:** `_load_line_config`, `_schedule_runs_to_days`. **Likely consequence:** unusable capacity dates. **Possible direction:** typed line-unit capacity and explicit operation routing (make, cool, pack, repack) per SKU.

## 15. Graham is actual-capable but scheduler-incompatible

**Problem:** 31012 correctly packs from ingredient 31011, yet scheduler/requirements demand a batch-type BOM component. **Why it matters:** order planning flags a valid factory repack as unschedulable. **Where:** `product_bom`; `_simulated_allocation`; `production_requirements`; standalone placeholder repack route. **Likely consequence:** manual planning outside the system. **Possible direction:** first-class repack/pack-only routing using source product, pack rate, packaging, and line assignment.

## 16. Schedule and actual never reconcile

**Problem:** no relation from a ledger event to schedule row and no automatic status/variance update. **Why it matters:** a confirmed/completed plan cannot be proven from the ledger. **Where:** `production_schedule` versus transaction schema/write paths. **Likely consequence:** stale plans and manual comparison. **Possible direction:** run identity, actual links, explicit match/split/merge rules, and variance audit.

## 17. Date semantics remain transitional

**Problem:** `timestamp`, `occurred_at`, `business_date`, and `created_at` coexist; code comments and actual conversion disagree. **Why it matters:** entries can land on the wrong operating day and corrections are ignored. **Where:** Phase 1 migration; history/calendar/activity queries. **Likely consequence:** daily totals/certifications are unreliable at boundaries. **Possible direction:** complete migration to typed effective occurred/business date, document plant cutoff, retain legacy field only for audit.

## 18. No general idempotency

**Problem:** duplicate client retries can repeat a commit. **Why it matters:** write receipts may be lost during a network timeout even when the DB committed. **Where:** all primary writes; `IDEMPOTENCY_KEY_PLAN.md` is unimplemented. **Likely consequence:** duplicate inventory. **Possible direction:** client idempotency keys, request hash, unique receipt record, deterministic replay response.

## 19. Order readiness is not allocation

**Problem:** fulfillment check compares each line/order against the same global stock and can include services; no reservation across demand. **Why it matters:** multiple orders can all appear ready for the same pounds. **Where:** `fulfillment_check`, order dashboard. **Likely consequence:** overpromising and schedule distortion. **Possible direction:** allocation/reservation model or clearly label check as nonexclusive availability; exclude services.

## 20. Production-demand export can overstate work

**Problem:** matrix uses full ordered pounds rather than remaining, and cancellation filters are incomplete. **Why it matters:** planners can schedule already-shipped/cancelled volume. **Where:** `export_orders_matrix`, workbook helpers. **Likely consequence:** overproduction. **Possible direction:** derive from a single canonical remaining-demand view and show source/status/as-of metadata.

## 21. Integrity score is coarse and partially raw

**Problem:** score subtracts per failed category, not affected records; several checks use raw status/lines and narrow conditions. **Why it matters:** 85/100 can hide hundreds of defects; the void count already undercounted effective voids. **Where:** `audit_integrity`. **Likely consequence:** false confidence. **Possible direction:** correction-aware checks, counts/rates/aging, severity by impact, coverage statement, authenticated sensitive detail.

## 22. Legacy views/tables/endpoints remain callable

**Problem:** old views use raw rows and obsolete enum strings; empty BOM tables remain; five legacy dashboard routes expose them. **Why it matters:** future code/AI may select a plausible but dead source. **Where:** schema views, `/dashboard/inventory|low-stock|today|lots|production`, `boms`/`bom_lines`. **Likely consequence:** zero or wrong reports and duplicated architecture. **Possible direction:** mark deprecated in schema/API, migrate consumers, then remove or replace with compatibility views over effective data.

## 23. Startup migration failures do not fail health

**Problem:** numerous schema/catalog mutations run at application startup and log nonfatal exceptions. **Why it matters:** a healthy API can run against partially migrated schema. **Where:** `main.py:startup`. **Likely consequence:** environment drift and runtime query failures. **Possible direction:** external, ordered, transactional migrations with an application-required schema version gate.

## 24. Public dashboard reads expose operational detail

**Problem:** inventory, production, shipments, receipts, lot detail, search, notes, and integrity are unauthenticated. Public dashboard reads now also expose entry timestamps through `/dashboard/api/activity/daily-entries`, while `/records/late` remains API-key protected. **Why it matters:** factory/customer/supplier operations may be commercially sensitive. **Where:** `/dashboard/api/*` GET and `/audit/integrity`. **Likely consequence:** unintended disclosure. **Possible direction:** authenticated dashboard/session or a deliberately scoped public read model; remove customer-level detail from unauthenticated diagnostics.

## 25. API/schema/instruction version drift and contract mismatch

**Problem:** application 3.1.1, main schema 3.4.0, instructions 3.7.0; instructions require behaviors backend does not enforce (supplier-lot day rule, simplistic “no BOM” pack). **Why it matters:** a GPT may promise safeguards the API lacks. **Where:** `main.py`, `openapi-gpt-v3.yaml`, GPT instruction files. **Likely consequence:** unsafe assumptions and hard troubleshooting. **Possible direction:** release manifest tying code/schema/prompts/migrations, generated contract checks, semantic rule tests.

## 26. Operational metadata is incomplete live

**Problem:** 590 lots lack receipt time and 49 receipt-entry lots lack supplier-lot code. **Why it matters:** FIFO and recall provenance weaken. **Where:** live `lots`; integrity endpoint. **Likely consequence:** fallback FIFO ordering and manual trace research. **Possible direction:** source-aware completeness rules, controlled backfill with provenance, and block future receipt commits lacking required metadata.

## 27. Tests leave high-risk flows uncovered

**Problem:** normal receive/make/pack/standalone ship, scheduler, and recursive trace lack broad direct/end-to-end coverage. **Why it matters:** core physical behavior can regress while the suite passes. **Where:** `tests/`. **Likely consequence:** production-only discovery. **Possible direction:** transaction-level fixtures, invariant/property tests for mass/effective state, browser/API contract tests, and recall scenarios.

## 28. Fallback sample data can be mistaken for actuals

**Problem:** process-flow and Sankey render samples on API failure. **Why it matters:** an outage may produce a plausible factory story. **Where:** `process-flow.html`, `sankey.html`. **Likely consequence:** decisions on fictional numbers. **Possible direction:** fail closed with an unmistakable unavailable state; keep demos behind explicit demo mode.

## 29. Production restrictions are surfaced but not enforced end-to-end

**Problem:** formula lookup and `/make` return `production_warning`, but product search, pack, inventory, dashboard, and planning paths omit it; direct commit has no acknowledgment requirement. The six SS Classic #9 finished rows carry the same note, yet `/pack` ignores it. **Why it matters:** an operator can still start or pack restricted production without seeing or proving compliance. **Where:** `build_production_warning`, `make`, `get_batch_formula`; omissions in search/dashboard/pack; Floor prompt rule. **Likely consequence:** the Kosher Ignition requirement depends on a compliant GPT workflow rather than a system invariant. **Possible direction:** typed/versioned production requirements, broad point-of-use propagation, required authenticated acknowledgment before commit, and a run-level compliance event.

## 30. The new kosher family is not reproducible or schedulable from repository structure

**Problem:** product/formula rows 283–290 were added directly to production with no migration/seed file; all eight lack line assignments and the six finished goods lack `product_bom`. **Why it matters:** a fresh environment cannot reconstruct the live catalog from the repository, and the API scheduler cannot route the family. **Where:** live `products`/`batch_formulas`; absence from migrations, `product_bom`, `product_line_assignments`, and `dashboard_config.json`. **Likely consequence:** environment drift, missing inventory panels, and manual planning. **Possible direction:** checked-in idempotent catalog migration/snapshot plus approved BOM, line, dashboard, and scheduler mappings.

---

# Part 15 — Graham Cracker Crumb Audit

## Answers to the ten audit questions

### 1. What Graham products exist?

**CONFIRMED LIVE:** SKU 31011 “Graham Cracker Crumbs – 50 LB” and SKU 31012 “Graham Cracker Crumbs – 10 LB.” The first is an active ingredient/resale source; the second is an active finished/copack product. No separate Graham batch product is required for the current 1:1 repack flow.

### 2. Where are they defined?

They are rows in `products`, identified by `odoo_code`. Dashboard membership is separately configured in `dashboard/dashboard_config.json`; name classification is in `dashboard/dashboard.js`; `no_production` invariants are in migration 038; 31012 repack notes also appear in order-matrix logic and tests. The standalone planner embeds separate copies of both catalog entries.

### 3. How are 50-lb bags represented?

31011 has `type='ingredient'`, `uom='50 lb bag'`, `case_size_lb=50`, `no_production=true`. Receipts add pounds to product-specific lots with supplier-lot/BOL metadata. It can ship directly or be a negative source line/ILC row in `/pack`.

### 4. How are 10-lb cases represented?

31012 has `type='finished'`, `uom='10 lb case'`, `case_size_lb=10`, `is_copack=true`, `no_production=false`, CNS/house-label metadata. Pounds on hand divided by 10 yields cases. Its `product_bom` names 10 lb of 31011, one SKU 21003 clear bag, and one SKU 21011 Box 24.

### 5. Is 10-lb packing treated as production?

Yes. It is transaction type `pack`, not `make`. That is an actual-production type for calendar/history. The source and target pound movements are equal. It is not a batch-production event despite a current generic Graham “Made · batches” placeholder in calendar family definitions; ordinary 31012 activity appears under Packed.

### 6. Where is it recorded?

One `transactions(type='pack')` header, positive `transaction_lines` for 31012 output lot, negative lines for 31011 source lot(s), and ILC consumption rows for those source lots. `main.py:pack` creates them. There are no packaging-material lines from `product_bom` under normal current behavior.

### 7. Can the dashboard retrieve it?

Yes. `/dashboard/api/production` retrieves positive effective posted 31012 pack lines. The finished-goods inventory endpoint includes exact 31012 via `dashboard_config.json`. The ingredient panel includes 31011 under Bulk Crumbs.

### 8. Does the Production Calendar show it?

Yes in the current deployed code. The audited 2026-08-10 activity displayed 4,200 lb / 420 cases. Current family logic recognizes “Graham,” and the layer-2 detail includes product name and SKU.

### 9. If it did not previously, why?

The ledger and API had the event, but the older browser summary recognized/rendered only Granola/Coconut. Graham was categorized as `other` and discarded by the renderer. This was a presentation-classification defect, fixed on 2026-08-11. A similar defect can recur for future families or renamed products because current classification is still string-based.

### 10. What changes are required for it to appear correctly and be fully represented?

For **today's calendar visibility**, none: it already appears. To make visibility robust, add a durable family=`graham`, operation=`repack`, display-unit=`case`, case size, and line/routing classification to backend product metadata; make the API return typed categories and a mandatory unclassified bucket; use effective business date.

To represent the **complete physical workflow**, model a pack/repack run with source bags/pounds, finished cases/pallets, packaging/label products and lots, line, operator/crew, stage timestamps (dump, erect cases, label, pack), expected versus actual output, scrap, and schedule link. `/pack` should enforce/version the 31011 -> 31012 routing and consume the bag/box/label BOM. These changes are ledger architecture changes, not just dashboard changes.

## Graham end-to-end reality

```text
CURRENT FACTUAL MODEL
Receive 31011 lot (+lb)
  -> Pack transaction
     - 31011 source lot pounds
     + 31012 finished lot pounds
     + ILC source lineage
  -> Ship 31012 (-lb)
  -> calendar derives cases = output lb / 10

PHYSICAL STEPS NOT CAPTURED
Open/dump 50-lb bags
  -> select/consume clear bag and box/label lots
  -> erect case
  -> fill/seal/label
  -> palletize, record scrap/labor/line
```

---

# Part 16 — File and Code Map

Paths are repository-relative.

| File | Purpose and important implementation | Related data/features |
|---|---|---|
| `main.py` | Entire FastAPI service. Startup/pool/auth; Pydantic requests; product resolution; `receive`, `ship`, `make`, `pack`, `adjust`; `build_production_warning`; corrections/certifications; trace/history; customer/orders; dashboard APIs; exports; scheduler; integrity. | Nearly every live table and API |
| `tests/schema/schema.sql` | Current dumped schema used by tests; includes migration 039 append-only/current-view/certification structures and migration 040 pack-format structure. | All database entities; no catalog data |
| `migrations/037_sales_order_flags.sql` | Adds dashboard readiness flags independent of order status. | `sales_order_flags` |
| `migrations/038_add_no_production_products.sql` | Marks 26 resale/nonproduction products and asserts 31012 remains producible. | `products.no_production`, Graham |
| `migrations/039_trial_timestamp_integrity.sql` | Phase 1 immutable creation timestamps, occurred/business dates, corrections, current views, certifications. | Core ledger/audit |
| `migrations/dry-runs/039_phase1_timestamp_integrity_dry_run.sql` | Deployment rehearsal/validation for Phase 1. | Integrity rollout |
| `migrations/040_product_pack_format.sql` | Adds constrained nullable finished-pack classification. | Calendar granola formats |
| `archive/migrations-applied/sql/*.sql` | Applied historical evolution: notes/scheduler, aliases, search, shipments, data repairs, case sizes, numeric precision, services, standalone shipment backfill, product cleanup, copack, dropped bake line. | Historical evidence only |
| `archive/migrations-applied/factory_ledger_reconciliation.sql` | Earlier reconciliation snapshot/script. | Historical repairs |
| `dashboard/index.html` | Main dashboard shell, tabs, panels, modals, navigation, and cache-versioned assets; current Activity order is Daily Entries, Shipping, Receiving and current references are JS v28/CSS v18 (Daily Entries minimum v27+/v17+). | Operations/activity/notes/orders |
| `dashboard/dashboard.js` | Main UI data fetch, production family/format summary, drill-down, inventory/activity/notes/orders/readiness/pallet/search behavior; Daily Entries rendering and four-row Shipping/Receiving expanders. | `/dashboard/api/*`, sales APIs |
| `dashboard/dashboard.css` | Main responsive/theme styles, calendar/family colors, Daily Entries states, and activity overflow/expander styling. | Presentation only |
| `dashboard/dashboard_config.json` | Exact names/groups and pallet assumptions for FG, batches, ingredients. | Main inventory visibility |
| `dashboard/mini-calendar.js`, `dashboard/mini-calendar.css` | Open-sales-order requested-ship-date calendar. | Sales schedule, not production |
| `dashboard/pallet-calculations.js` | 10-/25-lb case-size inference and line/order pallet math. | Orders/pallet display |
| `dashboard/process-flow.html` | Standalone recent-event conceptual line/stage view; hardcoded classifications/workers; sample fallback. | make/pack history |
| `dashboard/sankey.html` | Standalone material-flow visualization over make/ship history; omits pack; sample fallback. | history/customer flow |
| `dashboard/traceability.html` | Trace search, graph, recent-history indexes, completeness UI. | `/trace/*`, history |
| `dashboard/scheduler/seven-wells-production-board.html` | Self-contained localStorage planner/catalog/capacity engine. | Hypothetical plan only |
| `openapi-gpt-v3.yaml` | Main Custom GPT Action schema, exactly 30 operations, v3.4.0; formula/make warning descriptions; all descriptions at or below the editor's 300-character limit. | Main conversational client |
| `gpt-instructions-v3.md` | Main GPT operational instructions, v3.7.0. | Search/preview/commit behavior |
| `gpt-configs/schemas/openapi-floor.yaml` | Floor/Fulfillment Action schema, 22 operations, v4.1.0; warning relay contract and 300-character-safe descriptions. | Operator physical workflows |
| `gpt-configs/sources/shared-rules.md` | Shared generated GPT rules, including receipt-anchored success. | GPT behavior |
| `gpt-configs/sources/floor-specific.md` | Floor-specific bilingual/workflow rules, including verbatim EN+ES production-warning relay before make commit. | Floor GPT |
| `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` | Generated deployable Floor instructions; rebuilt 2026-08-12 at 7,833 characters. | Live-refreshed Floor GPT artifact |
| `build_gpt_instructions.py` | Combines Floor instruction sources; warns at 7,500 characters and fails at 8,000. | Prompt build process |
| `scripts/propose_pack_format_backfill.py` | Validates and optionally applies exact SKU-to-format mapping; migration 040 alone does not populate it. | `products.pack_format` |
| `scripts/dump_prod_schema.sh` | Refreshes schema snapshot from production. | Schema/test alignment |
| `scripts/setup_test_db.sh` | Builds database test environment. | Test infrastructure |
| `scripts/daily-health-ping.sh` | External health ping helper. | Availability, not a job engine |
| `scripts/cleanup_void_reversals.sh` | Legacy void/reversal cleanup helper. | Historical void semantics |
| `tests/test_phase1_ledger_integrity.py` | Append-only/correction/certification/current-view/calendar regression tests. | Phase 1 |
| `tests/test_production_warning.py` | Six tests for formula/make warning presence, absence, optional Spanish, commit behavior, and whitespace suppression. | Production requirements/Kosher Ignition |
| `tests/test_daily_entries.py` | Three tests for same-day entry metadata and signed lines, late-entry/entered-date behavior, and exclusion of effectively voided transactions. | Daily Entries/timeliness |
| `tests/test_void_semantics.py` | Effective void and balance behavior. | void/current ledger |
| `tests/test_dashboard_production_calendar.py` | Current calendar API count/category inputs and corrections. | Production Calendar |
| `tests/test_orders_matrix_export.py` | XLSX export, notes, Graham repack behavior. | Planning export |
| `tests/test_pallet_calculations.js` | Pallet conversion and unknown-format tests. | Dashboard pallet utility |
| `tests/test_ship_order_service_line.py` | Mixed physical/service shipment behavior. | Order fulfillment |
| `tests/test_write_response_contract.py` | Write receipt/error envelope. | API/GPT reliability |
| `FACTORY_LEDGER_CHANGELOG.md` | Detailed feature/deployment ledger and regression rationale; rows 56–58 cover the kosher family, warning rollout, and editor description limit. | Historical deployment evidence |
| `CHANGE_LOG.md` | Additional code change history, including all 2026-08-12 implementation/deployment steps. | Historical context |
| `STATUS.md` | File classification/status as of 2026-08-05; useful but predates latest work. | Orientation only |
| `CONTEXT.md` | Older architectural handoff, materially stale. | Historical context only |
| `FOLLOWUPS.md` | Known owner/engineering follow-ups: customer addresses, 4xx consistency, auto-create duplicates, instruction headroom, sibling Railway service. | Unresolved work |
| `VOID_SEMANTICS_RUNBOOK.md` | Operational semantics for void migration/cleanup. | Ledger correction history |
| `IDEMPOTENCY_KEY_PLAN.md` | Proposed, not implemented, idempotency design. | Future safeguard |
| `docs/deployments/phase-1-ledger-trial.md` | Phase 1 rollout evidence/status. | Deployment history |
| `docs/incidents/2026-08-05-gpt-actions-dispatch.md` | GPT Actions incident analysis. | Client integration history |
| `audits/reports/*.md` | Earlier fabrication/dashboard/trace audits. Some findings have since been fixed. | Historical diagnostic evidence |
| `requirements.txt`, `runtime.txt`, `netlify.toml` | Python dependencies/runtime and static deployment configuration. | Operations/deployment |
| `archive/superseded-*` | Obsolete schemas/instructions. | Never use as current contract |

---

# Part 17 — System Flows

## Granola production (make/bake actual)

```text
Physical ingredients are weighed and granola is produced
  -> operator tells Main or Floor GPT / direct client
  -> searchProducts resolves exact batch SKU
  -> make(mode=preview)
     -> products.default_batch_lb + yield_multiplier
     -> batch_formulas component quantities
     -> effective FIFO lot availability
     -> non-empty verification_notes -> production_warning
     -> Floor GPT relays warning verbatim; operator acknowledges
  -> operator approval under GPT workflow
  -> make(mode=commit) / `main.py:make`
     -> transactions: one posted `make` header
     -> transaction_lines: +batch output lot, -ingredient lots
     -> ingredient_lot_consumption: consumed-lot edges
  -> ledger_current_* views expose effective state
  -> `/dashboard/api/production` sums positive output by converted date/SKU
  -> browser name-classifies Granola and divides by declared output unit
  -> Production Calendar shows rounded Granola batches and product detail
```

There is no separate oven, pan, stage, line, labor, or downtime write. The process-flow page's Granola stage is reconstructed after the fact.

For 90025/90026, this flow returns the Kosher Ignition note from both formula lookup and make preview. The warning does not create an oven-ignition fact, and a direct API client can skip the formula/preview/acknowledgment sequence.

## Coconut production

```text
Coconut sweetening/toasting operation
  -> resolve coconut batch SKU
  -> `/make` loads coconut formula
     -> inventory components consumed
     -> Water row excluded from inventory
     -> output = default_batch_lb * requested batches * yield_multiplier
  -> make transaction + signed lines + ILC
  -> effective posted output
  -> calendar API:
       made_unit = default_batch_lb * yield_multiplier
       pans = round(output_lb / made_unit)
  -> browser name-classifies Coconut
  -> Production Calendar shows Coconut pans and SKU detail
```

Toasted timing/cooling constraints seen in the standalone planner are not written into this flow.

## Granola packing

```text
Granola batch lot is ready
  -> resolve source batch and target finished/label/format SKU
  -> `/pack` preview
     -> cases * target/override case weight
     -> effective FIFO batch lots
     -> optional parent-formula add-ins
  -> `/pack` commit
     -> one posted `pack` transaction
     -> -source batch lot(s), +finished output lot
     -> ILC for source/add-ins
  -> effective FG on-hand and batch on-hand update
  -> calendar API sums positive FG pounds
  -> browser classifies Granola + `pack_format`
  -> cases = round(lb / case_size_lb)
  -> day tile + 10-lb/25-lb/bagged SKU drill-down
```

The pack event does not normally deduct bag, box, or label products from `product_bom`.

## Graham cracker crumb packing

```text
Received 31011 50-lb-bag lot exists
  -> `/pack` source=31011, target=31012, cases=N
  -> required/output = N * 10 lb
  -> -31011 lot pounds +31012 finished lot pounds
  -> ILC connects source lot to pack transaction
  -> `/dashboard/api/production` returns 31012 positive pack output
  -> `getProductCategory()` matches “Graham”
  -> packed cases = round(output_lb / 10)
  -> Graham day summary and SKU detail
  -> later `/ship` or `/sales/orders/{id}/ship` consumes 31012 lot
```

Missing links: bag/box/label consumption, physical substeps, and recursive supplier-lot -> 31012 customer recall traversal.

## Production Calendar

```text
transactions + transaction_lines
  -> Phase 1 latest corrections
  -> ledger_current_transactions / ledger_current_transaction_lines
  -> filter effective posted + type make/pack + positive output
  -> convert legacy UTC-assumed timestamp to New York date
  -> SQL group date + product + transaction type
  -> backend derives rounded batch/pan/case count
  -> JSON day.batches[] / day.finished_goods[]
  -> browser string family + pack_format grouping
  -> layer 1 day totals
  -> click day
  -> layer 2 product name/SKU details
```

## Receiving to inventory to supplier trace

```text
Delivery + BOL + supplier lot
  -> `/receive` preview/commit
  -> system lot + receive header + positive line
  -> product/lot effective on-hand
  -> Receiving activity table
  -> later make/pack negative line + ILC
  -> `/trace/supplier-lot/{code}` immediate usage/direct shipments
```

The last step is not a guaranteed recursive customer-exposure graph.

## Sales order to shipment

```text
Customer/order lines entered
  -> sales_orders + sales_order_lines (confirmed)
  -> readiness check reads global effective on-hand (no reservation)
  -> shipOrder preview
  -> commitShipOrder / `ship_order`
     -> shipments header
     -> per physical order line: posted ship transaction + negative lot lines
     -> shipment_lines + sales_order_shipments
     -> quantity_shipped_lb / line status updated
     -> order partial_ship or shipped
  -> shipping activity + order detail
```

Voiding the ledger transaction does not currently unwind the order/support branch.

## API production schedule

```text
confirmed/in-production/ready order demand
  -> subtract direct order shipped pounds
  -> allocate current FG inventory
  -> product_bom chooses one batch-type component
  -> allocate existing batch inventory
  -> ceil remaining pounds / default_batch_lb
  -> batch_formulas ingredient explosion
  -> compare ingredient effective on-hand
  -> working-day + labor + batches/day allocation
  -> suggest earliest/latest JSON
  -> confirm -> production_schedule rows

NO AUTOMATIC EDGE TO:
  `/make` or `/pack` -> actual ledger -> schedule completion/variance
```

## Standalone production-board plan

```text
CSV/sample/manual orders + embedded SKU routing + local FG/WIP/settings
  -> browser greedy labor/capacity algorithm
  -> 84-day local plan, flags, pins, overtime
  -> localStorage `sw_prodboard_v1`

NO API/DB WRITE AND NO FACTORY LEDGER ACTUAL
```

---

# Part 18 — Current State Versus Intended Design

| Topic | Current implementation | Apparent intended design | Gap / confidence |
|---|---|---|---|
| Ledger actual | Effective posted signed lines are current inventory/actual. | Immutable, correction-audited factual record. | Phase 1 is deployed, but some reads/support tables remain raw/mutable. **CONFIRMED FROM CODE.** |
| Event date | Calendar/history often use converted legacy `timestamp`. | `occurred_at` and plant `business_date` separate activity time/date from DB receipt. | New fields exist but are not end-to-end. **CONFIRMED FROM CODE.** |
| Daily certification | Append-only certification and late-record API exist. | Operators certify complete days and surface late arrivals. | Only 1900-01-01 smoke records; operational workflow absent. **CONFIRMED LIVE; intent STRONGLY INFERRED.** |
| Granola actual | Make batch, then pack finished SKU. | Traceable ingredient -> batch -> format/label finished product. | Packaging/stages/line/labor are missing; source relations are duplicated. **CONFIRMED FROM CODE.** |
| Kosher production requirement | Eight SS Classic #9 products carry an English ignition note; formula and make responses surface it, and the Floor GPT is instructed to relay it before commit. | Prevent production until an authorized person ignites the oven and preserve proof of compliance. | No Spanish note live, no backend acknowledgment gate, no run-level compliance record, no pack/dashboard warning, and no scheduler/dashboard catalog mapping. **CONFIRMED FROM CODE/LIVE; proof requirement NEEDS OWNER CONFIRMATION.** |
| Coconut actual | Make with excluded water and yield multiplier; display as pans. | Capture sweetening/toasting production in meaningful plant units. | “Pan” semantics and stages are not modeled. **UNCERTAIN / NEEDS OWNER CONFIRMATION.** |
| Graham actual | 31011 ingredient -> `/pack` -> 31012 finished at 1:1 lb. | Repack 50-lb received crumb into labeled 10-lb cases. | Core quantity exists; labels/boxes/stages/scrap do not. **CONFIRMED FROM CODE/LIVE.** |
| Calendar | Two-layer actual display for three name-derived families and granola formats. | Fast daily snapshot with detailed product drill-down. | Interaction exists; taxonomy/date/metadata are fragile, unknown families omitted. **CONFIRMED FROM CODE.** |
| Product hierarchy | Flat products plus names, parent link, BOMs, config. | Stable family/variant/SKU/label/format/routing hierarchy. | No normalized hierarchy. Intended depth is **STRONGLY INFERRED** from UI/business request. |
| Recipes | Mutable `batch_formulas` drive make. | Reproducible formula/yield per production event. | No version or mass reconciliation. **CONFIRMED FROM CODE.** |
| Packaging BOM | Packaging rows can exist in `product_bom`. | Pack consumes exact bag/box/label materials/lots. | Execution ignores these rows. Intended consumption **NEEDS OWNER CONFIRMATION**. |
| Trace | Immediate backward/forward queries and graph UI. | Reliable one-step-back/one-step-forward recall. | Not recursive/correction-consistent; completeness badge overstates assurance. **CONFIRMED FROM CODE.** |
| Orders | Mutable workflow plus ledger shipment and support tables. | Fulfillment state agrees with physical shipments. | Void cannot unwind order state; readiness is nonexclusive. **CONFIRMED FROM CODE.** |
| Scheduling | API DB plan and separate local browser plan. | One capacity-aware plan tied to demand and actuals. | Two models, divergent capacities, no reconciliation. **CONFIRMED FROM CODE.** |
| Resale products | `no_production` flags selected products. | Do not schedule internal production for pass-through goods. | Scheduler/calendar/write paths do not uniformly enforce it. **STRONGLY INFERRED** from migration comments. |
| Production lines | Four capacity rows and product assignments. | Schedule and measure actual output by line/work center. | Actual transactions have no line. **STRONGLY INFERRED.** |
| Operator controls | Shared key + GPT preview/receipt rules. | Safe conversational operations with attributable approvals. | Prompt discipline is not identity/security/idempotency. **CONFIRMED FROM CODE.** |
| Odoo | SKU codes/names reflect Odoo; no integration. | Possibly catalog/order synchronization. | No current evidence of desired sync direction. **UNCERTAIN / NEEDS OWNER CONFIRMATION.** |
| Integrity | Eight-check public score and focused tests. | Continuous trustworthy data-health monitoring. | Coarse/raw checks and missing domains. **CONFIRMED FROM CODE.** |
| Dashboard process/Sankey | Reconstruct conceptual flows; sample fallback. | Management understanding of material/process flow. | Incomplete/event-capped and sometimes hypothetical. **CONFIRMED FROM CODE.** |

## How to use these labels in change discussions

- Do not describe an intended design as current behavior.
- Do not treat an old audit statement as current without verifying it; standalone shipments, effective calendar reads, Graham visibility, and line correction-based lot operations have changed.
- A confirmed code fact can still be a bug relative to business reality.
- A live count is an as-of snapshot; always attach the observation time.
- When owner intent is uncertain, present alternatives and identify the schema/write/read consequences before recommending one.

---

# Part 19 — Questions for the Owner

These questions cannot be answered confidently from the repository and materially affect architecture.

## Product terminology and catalog

1. Is one Granola “batch” always one `/make` requested batch, one oven load, one mix, or `default_batch_lb * yield_multiplier` pounds? Can those differ?
2. For Coconut, what exactly is one “pan,” and are sweetened and toasted pans counted by the same physical rule?
3. Should product family, customer label, brand, pack format, routing, and bilingual production requirements be governed master data? Which system owns them—Factory Ledger or Odoo? What is the approved Spanish Kosher Ignition wording?
4. What is the approved correction for SKU 90008: 25-lb add-in formula, 384.52-lb batch output, or a nested Classic-batch recipe? Should earlier 25-lb makes remain historically valid?
5. What should active bulk-per-pound finished SKUs 70004, 70006, 70013, and 70016 use as case/unit semantics, or should they be explicitly per-pound/noncase products?

## Production workflow

6. Must the ledger track boxes, bags, and labels as consumed inventory lots during packing, or is pound-only finished output the intended scope?
7. For Graham 31012, which exact case/label/box materials and quantities are physically consumed, and is Box 24 (SKU 21011) the correct current box rather than similarly named packaging SKUs?
8. Do case erection, cleaning, sanitation, changeover, cooling, toasting, scrap, and rework need factual event records or only planning/notes?
9. Should `/make` and `/pack` hard-block products with incompatible type/routing/no-production flags, and should a non-empty production requirement require an authenticated acknowledgment? Do the notes copied to finished SKUs 70013–70018 mean packing must warn too, or is Kosher Ignition only a make-time control?
10. Are source and target pounds expected to balance for every pack, and how should normal giveaway/moisture/scrap be recorded?

## Dates, audit, and traceability

11. What is the plant business-day cutoff and authoritative timezone for legacy events? Should historical `timestamp` values all be interpreted as UTC?
12. Who certifies a day, from what source, and what operational response is required for a late record after certification?
13. What traceability standard is required: immediate one-up/one-down, full recursive supplier-to-customer, packaging-lot recall, or FDA/third-party certification?
14. For legacy/found/production-output lots, when is `received_at` required? Should the 590 missing values be backfilled, exempted by source type, or both?

## Dashboard behavior

15. Which production families must appear beyond Granola, Coconut, and Graham, and should an `Other/Unclassified` group always be visible rather than “No production”?
16. Should calendar counts round to whole units per product, show fractional units, or use the originally entered cases/batches when available?
17. Is the current Graham detail label “Packed · labels” meaningful, or should it be “10-lb cases/repack”?
18. Are the exact configured dashboard product lists intentional curation, or should panels automatically include all active products by governed category?
19. Should API/sample fallback visualizations ever render in the production dashboard, even with a banner?

## Orders and inventory

20. When a shipment is voided, should the order automatically reopen and all support rows reverse, or must an operator approve a separate fulfillment correction?
21. Does “ready” mean nonexclusive stock availability, physically allocated stock, completed packing, or a manual floor signal?
22. Should the order matrix represent total ordered demand or remaining unshipped/uncancelled demand?
23. Is automatic customer creation during order entry acceptable, and what system owns customer/address cleanup?

## Scheduling

24. Which scheduler is intended to survive: the database/API scheduler, the standalone production board, or a merger of both? Should the new SS Classic #9 family receive `product_bom` and line assignments immediately?
25. Which capacity set is authoritative where the two disagree (for example Pouch 7,500 versus 8,500 bags/day), and who owns updates?
26. Should schedules include separate make, cool/toast, pack/repack, cleaning/changeover, and material-prep operations?
27. How should actual make/pack events match planned work when a run covers several orders, products, lots, or dates?

## Security and governance

28. Which roles should be allowed to view inventory/customer data, post floor events, acknowledge production restrictions/ignition, edit catalog/BOM, correct/void history, run admin SQL, and certify days?
29. Is exposing operational dashboard reads outside authentication intentional?
30. What retention/deprecation policy should apply to backup tables, legacy views, superseded GPT artifacts, and sample data?

---

# Part 20 — How an AI Should Reason About the Factory Ledger

## The mandatory reasoning chain

For every architectural or discrepancy question, trace all six layers:

```text
BUSINESS PROCESS
  What physically happened or should happen?
        ↓
DATA CAPTURE
  Who/what calls which preview and commit? What cannot be captured?
        ↓
DATA MODEL
  Which product, lot, transaction, line, ILC, order, support, or plan rows exist?
        ↓
BUSINESS LOGIC
  What conversion, allocation, rounding, yield, category, or state rule applies?
        ↓
QUERY / API
  Raw or effective? Which date? Which joins, filters, limits, and response fields?
        ↓
DASHBOARD PRESENTATION
  What client grouping, exact list, fallback, rounding, or omission changes meaning?
```

Never jump from “not visible” to a CSS/UI fix. First determine whether the actual event exists and is modeled correctly.

## Core rules for a future Custom GPT

1. **Separate fact, derivation, plan, and intent.** Say “the ledger records,” “the dashboard derives,” “the schedule proposes,” or “the owner appears to intend.”
2. **Use effective state.** Prefer `ledger_current_transactions` + `ledger_current_transaction_lines`, `effective_status='posted'`. Flag raw paths explicitly.
3. **Name the unit and formula.** Pounds, cases, batches, pans, bags, and pallets are not interchangeable. State denominator and rounding.
4. **Anchor time.** Distinguish occurred time, business date, created/received time, and schedule date. Treat live counts as as-of facts.
5. **Trace lots for provenance.** Product total answers quantity, not which supplier/output lot or customer exposure.
6. **Do not infer physical stages.** A `pack` proves signed material conversion, not label/box/labor/stage execution unless those records are added.
7. **Treat names/config as fragile.** A dashboard family match is not a catalog invariant.
8. **Keep schedule and actual separate.** No schedule row proves production; no actual currently closes a schedule.
9. **Preserve history.** Prefer append-only correction and versioned metadata over destructive edits when a proposed change alters past interpretation.
10. **State uncertainty.** If business meaning is absent, ask an owner question rather than inventing it.
11. **Never expose credentials.** Do not quote the embedded shared secret; recommend remediation by location only.
12. **For operational writes, use current GPT safety rules.** Search/resolve, preview, receive operator approval where required, commit, and claim success only from the returned receipt. This document is knowledge, not authorization to mutate data.
13. **Treat `production_warning` as a stop-and-relay control.** Quote `verification_notes` and optional `_es` verbatim before make commit, obtain explicit acknowledgment, and never infer that acknowledgment or ignition occurred from the warning itself. Recognize that direct API enforcement is still missing.

## Diagnostic playbooks

### “Why isn't this production showing on the dashboard?”

Check in order:

1. Was a `make` or `pack` commit actually created, or only a schedule/note/GPT statement?
2. Does transaction history show the expected positive product/lot line?
3. Is effective status posted, and did a correction change product/lot/quantity/date?
4. Is the event inside the queried date after timezone/business-date rules?
5. Does `/dashboard/api/production` return it?
6. Does the product have required `default_batch_lb`, yield, or case size?
7. Does `getProductCategory` recognize the name, and for packed granola is `pack_format` set?
8. Is it omitted by exact dashboard configuration or frontend grouping?

Classify the fix: capture/ledger, catalog metadata, API/effective-date query, or presentation. Graham's historical failure was presentation; a missing transaction would not be.

### “Can we add a new product?”

First define SKU identity, type, unit/case size, batch size/yield, production family, brand/label/customer, no-production/service/copack flags, parent/routing, execution formula, product/packaging BOM, line assignment/capacity, pack format, verification/production requirements (including approved bilingual text), and dashboard placement. Then assess migration/backfill, GPT resolution/schema implications, and tests. Adding only a `products` row may make inventory possible but leave production, scheduling, calendar, export, warnings, and trace incomplete. The 70013–70018 family is the concrete example: parent/formula operation is possible, while dashboard inventory and scheduler mappings are absent.

### “What should I do with a production warning?”

Identify the product and exact response source. Preserve the English and optional Spanish notes byte-for-meaning; do not summarize away the requirement. Show them before commit, require an explicit operator acknowledgment, and only then submit the commit. After success, report the transaction receipt but do not claim the physical prerequisite (for example oven ignition) was performed—the system has no such event. If a warning was expected but search/dashboard/pack does not show it, call formula lookup or make preview and flag the surface gap rather than assuming no restriction exists.

### “Can we change how production is summarized?”

Identify whether the requested number is:

- a new factual field/event (ledger/schema/write-path change);
- a new interpretation of existing pounds (business logic/API change);
- a new stable classification (catalog/taxonomy change plus API/UI);
- only layout/drill-down of already typed API data (dashboard change).

Demand an example with SKU, raw pounds, desired unit, date, and expected rounding. Test historical and correction cases.

### “Where should this data be stored?”

- Physical quantity/provenance -> transaction/line/lot or append-only related consumption/event entity.
- Master definition/routing -> governed, versioned catalog/BOM tables.
- Mutable order workflow -> order tables, reconciled with ledger shipment facts.
- Plan/capacity -> canonical scheduling domain, explicitly not actual.
- Annotation/task -> notes/flags.
- Pure display preference -> frontend config/state.

Do not store a factual production stage only in a note or browser `localStorage`.

### “Would this change break historical records?”

Ask whether it changes product name, type, case size, batch size, yield, formula, source mapping, family, timestamps, status, or trace edges. Because many reports use current metadata, even a master-data edit can reinterpret old events. Prefer new effective-dated version/SKU or event snapshot; use ledger corrections for fact errors; create migration and before/after reconciliation queries.

### “Is this a dashboard change or ledger architecture change?”

- Event exists and API returns all stable semantics; only rendering is wrong -> dashboard.
- Event exists but category/conversion lives only in string rules -> catalog/API plus dashboard.
- Physical fact is absent (labels, boxes, line, scrap, stage, labor) -> ledger/write architecture first.
- Plan/actual relationship is missing -> scheduling + ledger integration.

### “What is the source of truth for this number?”

Answer with a chain, not a filename: effective transaction IDs/lines -> product/lot -> conversion metadata -> API function -> client aggregation. State raw versus effective, date field, filters, and as-of timestamp. If the number comes from sample/local plan, explicitly say it is not an actual.

### “Can we add a drill-down?”

Verify the layer-1 aggregate retains stable identifiers needed for layer 2: product ID/SKU, transaction IDs, lot IDs, date, operation type, and unrounded pounds. If not, extend the read model first. Do not reconstruct identity from display text. Ensure drill-down totals reconcile to the summary under one rounding policy.

### “Why do these totals disagree?”

Compare:

- effective versus raw status/lines;
- product total versus lot total;
- event/business/created/schedule date;
- pounds versus rounded cases/batches/pans/pallets;
- current versus event-time product metadata;
- make versus pack versus ship/receive/adjust inclusion;
- exact product list/family/pack-format filters;
- full history versus 100/500-row client caps;
- order shipped aggregate versus effective ship ledger;
- actual versus either scheduler.

Produce a small reconciliation table with each definition before deciding one side is wrong.

## Change-impact checklist

For any proposal, identify all affected layers:

- schema/migration and historical backfill;
- core write validation and receipt contract;
- effective correction/void behavior;
- balance and trace invariants;
- customer/order/shipment side effects;
- read endpoints and raw/effective migration;
- dashboard main panels and every standalone view;
- GPT Action operation cap/schema and generated instructions;
- GPT editor limits (8,000 instruction characters and 300 characters per OpenAPI description/summary) and live editor refresh status;
- production-warning propagation, bilingual text, acknowledgment, and run-level compliance evidence;
- scheduling/requirements implications;
- authentication/privacy exposure;
- tests, deployment order, cache versions, and dual changelogs required by `CLAUDE.md`.

## Minimum evidence for an architectural recommendation

An AI recommendation should contain:

1. **Current behavior** with file/function/table evidence.
2. **Business intent** labeled confirmed, inferred, or owner question.
3. **Root cause** at the earliest incorrect layer.
4. **Proposed invariant** stated independently of UI.
5. **Affected write/read/data/dashboard/GPT paths.**
6. **Historical migration and correction strategy.**
7. **Failure modes and validation tests.**
8. **Explicit plan-versus-actual and source-of-truth impact.**

The goal is not merely to make a number appear. It is to ensure that the number represents a captured physical fact, remains reproducible after correction/catalog change, and can be traced from factory action to decision-making display.

---

# Final Consistency Checklist

- Major ledger/master/support/plan tables and duplicated concepts documented: **yes**.
- Granola, Coconut, Graham, packaging, labels, customers, orders, inventory, labor/line/stage gaps documented: **yes**.
- Receive, make, pack, ship, adjust, found, corrections, lot operations, orders, and schedule write paths documented: **yes**.
- Canonical/legacy history, inventory, activity, trace, order, and dashboard read paths documented: **yes**.
- Main Dashboard tabs, cards/tables, calendar/drill-down, notes, orders, health/search, process-flow, Sankey, trace, and local planner documented: **yes**.
- Business formulas, inputs, output, source, rounding, and assumptions documented: **yes**.
- Production Calendar date, selection, categorization, conversions, exclusions, Graham fix, and layered design documented: **yes**.
- Schedule and actual separated without ambiguity: **yes**.
- Source-of-truth hierarchy and nonauthoritative sources identified: **yes**.
- Live point-in-time evidence separated from enduring code facts: **yes**.
- SS Classic #9 product IDs/SKUs, formula lineage, finished-product relationships, warnings, and current planning/dashboard omissions reverified against live data and code: **yes**.
- Production-warning response surfaces, non-surfaces, GPT relay contract, editor limits, and missing backend enforcement documented: **yes**.
- Current test count and schema snapshot coverage for migrations 039/040 reverified: **yes**.
- Known gaps include consequence and possible direction: **yes**.
- Graham ten-question mini-audit and full-model requirements completed: **yes**.
- Owner questions limited to material unresolved business/governance decisions: **yes**.
- AI operating guide traces business process through presentation and resists superficial fixes: **yes**.
- Shared credentials omitted from this document: **yes**.

# Architecture map (code locator)

Where each part of Factory Ledger lives, so a session can open the right code and tests without scanning the repo. Product behaviour and staff workflows: `docs/operating-model/01-architecture.md`. Safety rules: `CLAUDE.md`.

Pointer syntax: file::text means search that file for that exact text (a section banner or `def` line). Never trust a line number written down anywhere. Get current lines with `grep -n 'RECEIVE ENDPOINTS' main.py`, then read about 200 lines from there. `scripts/check_architecture_map.py` fails if any path, file::text anchor or test glob below stops existing, or if a new `main.py` section or root module is missing from this map.

## Runtime shape

- **Backend:** one FastAPI app in `main.py`, about 21.1k lines, deployed to Railway. Railway redeploys only when root `*.py`, `Procfile`, `requirements.txt`, `runtime.txt`, `railway.json` or `dashboard/dashboard_config.json` change (`railway.json` watchPatterns).
- **Feature modules:** files at the repo root expose `register_routes(app, api)`. Here `api` is the `main` module itself, so a feature module calls back into `main` as `api.<name>`. Before changing a `main.py` helper, run `grep -n 'api\.<name>' *.py`. Registration happens in `main.py::key_sessions.register_routes(app` and in the block near the end, `main.py::write_tickets.register_routes(app`.
- **Dashboard:** static HTML/JS in `dashboard/`, served by Netlify (`netlify.toml`, base `dashboard`). `dashboard/dashboard-runtime.js` picks the production or staging API.
- **Database:** Supabase Postgres. Changes go through `migrations/` (numbered NNN_name.sql files, reverts in `migrations/down/`). Older DDL still runs at startup, gated by marker rows: `main.py::# Connection pool (initialized on startup)`.
- **Custom GPTs:** schema `openapi-gpt-v3.yaml` (30-operation hard limit) and instructions `gpt-instructions-v3.md` (8,000-character limit), built with `build_gpt_instructions.py`. Floor and Office configs live in `gpt-configs/`.

## Modules

### Products, admin, health, audit
- Code:
  - `main.py::# PRODUCT SEARCH ENDPOINTS`, `main.py::# QUICK-CREATE PRODUCT ENDPOINTS`, `main.py::# PRODUCT VERIFICATION ENDPOINTS`
  - `main.py::# ADMIN PRODUCT & BOM MANAGEMENT`, `main.py::# REASON CODES ENDPOINT`
  - `main.py::# HEALTH & ROOT ENDPOINTS`, `main.py::# AUDIT / INTEGRITY CHECK ENDPOINT`
  - Supplies: `supplies.py` (`/supplies/inventory`, `/supplies/inventory/{product_id}/lots`, `/supply-requests` GET/POST/PATCH and the `SupplyRequestCreate` / `SupplyRequestUpdate` models; moved out of `main.py` in PR2). It is registered at `main.py::supplies.register_routes(app`, between the sales-order routes and `main.py::# SHIP ENDPOINTS`. It uses `api.verify_api_key`, `api.get_transaction`, `api.POSTED_LINES`, `api.fifo_lot_balances`, `api.ledger_quantity_unit` and `api.logger`. `main.py::def ledger_quantity_unit` stays in `main.py` because other dashboards and `exceptions_enforcement.py` use it. Inventory is read-only here; the supply-request writes touch only `supply_requests`, never transactions, lots or allocations.
- Tests: `tests/test_archived_products_dashboard*`, `tests/test_supplies.py`, `tests/test_health_v3.py`
- Risk: Medium; **High** for product flags used by ship or pack (case size, service lines)

### Inventory, lots, voids, history
- Code:
  - `main.py::# INVENTORY ENDPOINTS`, `main.py::# LOT ENDPOINTS`
  - `main.py::# ADJUST ENDPOINTS`, `main.py::# VOID TRANSACTION ENDPOINT`
  - `main.py::# TRANSACTION HISTORY ENDPOINTS`, `main.py::# LOT REASSIGNMENT ENDPOINTS`, `main.py::# FOUND INVENTORY ENDPOINTS`
  - `main.py::def void_transaction`
  - `lot_confirmation.py`, `supplier_lot_tickets.py`
- Rules: inventory sums only `transactions.status='posted'`. Never edit or delete transactions (see `VOID_SEMANTICS_RUNBOOK.md`). Never give pooled lots a single supplier lot.
- Tests:
  - `tests/test_void_*`, `tests/test_inventory_occurred_at.py`, `tests/test_lot_identity_047.py`
  - `tests/test_lot_confirmation.py`, `tests/test_batch_lot_balances_round_d.py`
  - `tests/test_phase1_ledger_integrity.py`, `tests/test_supplier_lot_tickets.py`
- Risk: **High**

### Receiving and expected receipts
- Code:
  - `main.py::# RECEIVE ENDPOINTS`, `main.py::def _receive_preview_core`, `main.py::def _receive_commit_core`
  - `expected_receipt_tickets.py`, `extraction.py`, `attachments.py`, `assistant_receive.py`
  - Prepare/commit tickets: `write_tickets.py`
- UI: `dashboard/receipt.js`, `dashboard/intake-logic.js`, `dashboard/entry.js`
- Tests:
  - `tests/test_receive_*`, `tests/test_expected_receipt*`, `tests/test_er_intake_logic*`
  - `tests/test_final_receive_receipts.py`, `tests/test_floor_gpt_receive_contract.py`
- Risk: **High** for commit or lot rules; Medium for preview or UI

### Production (make), BOM, kosher, shift reports
- Code:
  - `main.py::# MAKE (PRODUCTION) ENDPOINTS`, `main.py::def _make_commit_core`
  - `main.py::# BOM ENDPOINTS`, `main.py::# PRODUCTION REQUIREMENTS`, `main.py::# PRODUCTION DAY SUMMARY`
  - `oat_bran_substitution.py`, `kosher_tiers.py`, `shift_summary.py`, `shift_receipts.py`
- UI: `dashboard/runs.js`, `dashboard/shift-summary.js`, `dashboard/scheduler/`
- Tests:
  - `tests/test_production_*`, `tests/test_run_types.py`, `tests/test_oat_bran_*`, `tests/test_kosher_*`
  - `tests/test_recipe_corrections_078.py`, `tests/test_shift_*`
- Risk: **High** (ingredient consumption)

### Packing
- Code: `main.py::def _pack_preview_core`, `main.py::def _pack_commit_core` (inside the MAKE section), `main.py::def _post_prepared_inputs`
- Rule: add-in ingredients are deducted at pack time.
- UI: `dashboard/pallet-calculations.js`
- Tests: `tests/test_*packing_slip*`, `tests/test_assistant_pack_review_c.py`, `tests/test_pallet_calculations.js`
- Risk: **High** (consumption)

### Shipping
- Code:
  - `main.py::# SHIP ENDPOINTS`, `main.py::def _ship_order_commit_core`
  - Retry guard: `main.py::def _ship_order_retry_guard` (migration `migrations/088_ship_order_idempotency.sql`)
  - `ship_tickets.py`
- Tests: `tests/test_ship_*`
- Risk: **High**

### Sales orders and customers
- Code:
  - `main.py::@app.post("/customers"`, `main.py::def _create_sales_order_core`
  - Routes `/sales/*` and `/export/*` sit between `main.py::# REASON CODES ENDPOINT` and `main.py::# ADMIN PRODUCT & BOM MANAGEMENT`, with no section banner. Find them with `grep -n '^@app.*"/sales' main.py`.
  - `order_tickets.py`, `customer_tickets.py`, `sales_document_tickets.py`
- UI: `dashboard/so-list.js`, `dashboard/so-detail.css`, `dashboard/office.js`, `dashboard/office-logic.js`
- Tests:
  - `tests/test_sales_order_*`, `tests/test_order_*`, `tests/test_customer_tickets.py`, `tests/test_resolve_customer.py`
  - `tests/test_orders_matrix_export.py`, `tests/test_direct_order_close_guard.py`, `tests/test_so_list_actions_ui.js`, `tests/test_office_*`
- Risk: Medium; **High** when it touches allocations, ship or close
- Contract: `docs/order-create-contract.md`

### Traceability
- Code: `main.py::# TRACE ENDPOINTS`; trace tables come from migration 048
- UI: `dashboard/traceability.html`
- Tests: `tests/test_trace_*`, `tests/test_production_trace.py`
- Docs: `docs/traceability/`
- Risk: **High**

### Dashboard (Netlify)
- Shell: `dashboard/index.html`, `dashboard/dashboard.js`, `dashboard/shell.js`, `dashboard/dashboard-runtime.js`, `dashboard/ledger-ui.js`
- Pages: `dashboard/history.js`, `dashboard/entry.js`, `dashboard/office.js`, `dashboard/exceptions.js`, `dashboard/pin.js`
- Backend for the pages: `main.py::# DASHBOARD ENDPOINTS` and `/dashboard/api/*` routes (`grep -n '"/dashboard/api' main.py`)
- Notes panel: `notes_api.py` (the five `/dashboard/api/notes` routes and the `NoteCreate` / `NoteUpdate` models; first module moved out of `main.py`, PR1). It is registered at `main.py::notes_api.register_routes(app`, the spot the section used to occupy, before the `/dashboard` static mount. It uses `api.verify_api_key`, `api.get_transaction`, `api._record_actor_write`, `api._is_readonly_error`, `api.format_timestamp` and `api.logger`; the key allowlist (`main.py::# Dashboard notes (the dashboard's own list and CRUD)`) and `permissions.py` stay where they were. Every route needs a key (the GET too) and writes by a named person are attributed; see `docs/security/notes-endpoints.md`.
- Tests:
  - `tests/test_dashboard_*`, `tests/test_*_ui.js`, `tests/test_nav_links.js`
  - Notes: `tests/test_notes_characterization.py`, `tests/test_notes_auth.py`
  - Visual: `tests/visual/` (Playwright, `package.json`)
- Run JS tests with `node --test tests/<file>.js`.
- Risk: Low for wording or CSS; Medium for flows that write

### Authentication, keys, roles
- Code:
  - `main.py::def _authorize_api_key`, `main.py::def verify_api_key`
  - `permissions.py`, `key_sessions.py`, `device_pins.py` (phone PIN, migration `migrations/089_device_pins.sql`), `staging_safety.py`
- UI: `dashboard/key-session.js`, `dashboard/pin.js`
- CI: `.github/workflows/key-literal-guard.yml` runs `scripts/check_no_key_literals.py`
- Tests:
  - `tests/test_key_*`, `tests/test_roles_a2.py`, `tests/test_readonly_*`, `tests/test_named_actor_writes.py`
  - `tests/test_actor_attribution.py`, `tests/test_notes_auth.py`, `tests/test_staging_safety.py`
- Docs: `docs/operating-model/06-access-reference.md`, `docs/named-actor-writes.md`
- Risk: **High**

### FL Assistant (in-app chat) and tickets
- Code:
  - `fl_assistant.py` (routes `/assistant/*`), `assistant_*.py`
  - `write_tickets.py` (`/…/prepare`, `/tickets/{ticket}/commit`, `/receipts`), `ticket_actions.py`
  - `resolution.py`, `resolution_receipts.py`, `exceptions_enforcement.py`
- UI: `dashboard/fl-assistant.js`, `dashboard/assistant-display.js`
- Tests: `tests/test_assistant_*`, `tests/test_fl_assistant*`, `tests/test_write_tickets*`, `tests/test_resolution*`, `tests/test_exceptions_*`
- Docs: `docs/resolution.md`
- Risk: Medium; **High** when the commit path changes

### Database and migrations
- Code: `migrations/` (numbered, idempotent, never edit an applied file), `migrations/down/`, `migrations/dry-runs/`
- Test schema: `tests/schema/schema.sql` (refresh with `scripts/dump_prod_schema.sh`)
- Tests:
  - `tests/test_startup_migrations.py`, `tests/test_*_0[0-9][0-9].py`
  - `tests/test_db_pool_wait.py`, `tests/test_audit_savepoints_lotcodes.py`
- Risk: **High**

### Deployment and operations
- Config: `railway.json`, `Procfile`, `netlify.toml`, `requirements.txt`
- Docs and checks:
  - `DEPLOYMENT.md`, `docs/deployments/`, `docs/staging.md`
  - Staging checks: `scripts/check_*_staging.py`, `scripts/seed_staging.py`
- Tests: `tests/test_server_keepalive.py`, `tests/test_health_v3.py`, `tests/test_seed_staging.py`, `tests/test_key_literal_guard.py`
- Risk: **High**

## Tests

- **Python:** `scripts/run_tests.sh tests/test_x.py -x -q`. It uses the `.venv-test` interpreter and the LOCAL database `factory_ledger_test` (`scripts/setup_test_db.sh`). `tests/conftest.py` refuses production hosts. Tests marked `db` skip when no test database is set.
- **Choosing tests:** run the module's tests above first, then `grep -l '<function or route>' tests/` for anything else that calls the changed code. The full suite (about 6,300 tests in each of two modes) is for High-risk changes and pre-merge only.
- **API contract snapshot:** `tests/test_route_contract.py` compares the live route table and a normalized OpenAPI document with `tests/fixtures/route_contract/`. It covers route order, methods, paths, names, parameters, dependencies, status codes and response models. It needs no database. A change that is not meant to alter the API must leave it green without regenerating. Regenerate only for a deliberate API change (`FL_UPDATE_ROUTE_CONTRACT=1`), and review the fixture diff line by line.

## Moving code out of main.py

The plan is to extract one section per PR. Done so far: Notes (`notes_api.py`, PR1), Supplies (`supplies.py`, PR2). For each extraction:

- **New module at the repo root.** Railway's `watchPatterns` only watch `/*.py`, so code in a subfolder would merge but never deploy.
- **Same interface as the existing modules.** Expose `register_routes(app, api)` and call it from `main.py` at the exact spot the section occupied, so route order does not change. Routes under `/dashboard/*` must register before the static mount.
- **Never import main.** Reach it as `api.<name>` at call time. `tests/test_feature_module_imports.py` enforces this and also checks that every `register_routes` is actually called.
- **Prove it changed nothing.** `tests/test_route_contract.py` must stay green without regenerating. Also run the section's characterization tests (Notes: `tests/test_notes_characterization.py`; Supplies: `tests/test_supplies.py`).
- **Update this map in the same PR**, then run `python3 scripts/check_architecture_map.py`.

## Where not to search by default

These hold evidence and history, not code. Exclude them from routine searches and open them only when the task is about that evidence:

- `docs/validation/`, `docs/design/audit/`, `audits/`, `archive/`, `tests/visual/` output
- `CHANGE_LOG.md` (about 1 MB)
- `FACTORY_LEDGER_CHANGELOG.md` (about 450 KB; search it, don't read it whole)

Example: `git grep -n 'supplier_lot' -- ':!docs' ':!audits' ':!archive' ':!*.md'`

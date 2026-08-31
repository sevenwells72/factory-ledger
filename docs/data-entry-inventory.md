# Data-Entry Path Inventory

Generated 2026-08-25 against `main` @ 65bca82 (main.py, 14,823 lines — includes PR #18 `feat/inventory-occurred-at`).
Schema sources: office GPT `openapi-gpt-v3.yaml` **v3.5.0** (30 ops), floor GPT `gpt-configs/schemas/openapi-floor.yaml` **v4.1.0** (22 ops).
Read-only audit; line numbers reference main.py at the commit above.

Legend — **GPT schema**: `office` = openapi-gpt-v3.yaml, `floor` = floor schema, `both`, `neither`. `[D]` = also on `DASHBOARD_KEY_ALLOWLIST` (callable with the scoped dashboard key; everything else needs the master `X-API-Key`).

## Write endpoints

| Endpoint | Table(s) written | Business event | GPT schema | Required fields | Lot code |
|---|---|---|---|---|---|
| POST /products/resolve | **none** (read-only despite POST) | other — bulk name resolution | office | `names[]` | n/a |
| PATCH /lots/{lot_code}/supplier-lot | lots; lot_supplier_codes (commingled only) | correction — lot metadata | both | `supplier_lot_code` (lot_code in path) | required (path) |
| PATCH /lots/{lot_id}/rename | lots | correction — fix lot code | floor | `new_lot_code` | required (new code in body; lot by id) |
| POST /receive (+ /receive/preview, /receive/commit `[D]`) | lots (find-or-create + update), transactions, transaction_lines, expected_receipts (auto-close), lot_supplier_codes | receipt | both | `product_name`, `cases`, `case_size_lb`, `shipper_name`, `bol_reference` | **optional** — auto-generated `{yy-mm-dd}-{SHIP}-{seq}` if omitted |
| POST /suppliers `[D]` | suppliers | PO (supplier master) | neither | `name` | n/a |
| POST /expected-receipts `[D]` | expected_receipts | PO (expected receipt; explicitly not a PO system) | office | `supplier_name` (must resolve), `expected_qty`; one of `product_id`/`product_name` | n/a |
| PATCH /expected-receipts/{id} `[D]` | expected_receipts | PO | neither | ≥1 field (all optional) | n/a |
| POST /supply-requests `[D]` | supply_requests | other — supplies queue | neither | `requested_by`; exactly one of `product_id`/`item_text` | n/a |
| PATCH /supply-requests/{id} `[D]` | supply_requests | other — supplies queue | neither | `status` (only `"done"`) | n/a |
| POST /ship (+ /ship/preview, /ship/commit) | transactions, transaction_lines, shipments, shipment_lines; customers (auto-create); sales_order_allocations (expire/shrink) | ship (standalone) | both | `product_name`, `quantity_lb`, `customer_name`, `order_reference` | **optional** — pins plan to that lot; FIFO if omitted; never creates lots |
| POST /make (+ /make/preview, /make/commit) | lots (output), transactions, transaction_lines, ingredient_lot_consumption | batch | both | `product_name`, `batches` (+`confirmed_sku=true` when SKUs share a BOM) | output **optional** — auto `B{yymmdd}-{seq}`; ingredient lots FIFO, `ingredient_lot_overrides` optional |
| POST /pack (+ /pack/preview, /pack/commit) | lots (FG target), transactions, transaction_lines, ingredient_lot_consumption; sales_order_allocations (expire/shrink) | pack | both | `source_product`, `target_product`, `cases` (+`case_weight_lb` if product lacks case_size_lb) | target **optional** — inherits first consumed batch lot's code; source `lot_allocations` optional (FIFO default) |
| POST /adjust (+ /adjust/preview, /adjust/commit) | transactions, transaction_lines | adjustment | both | `product_name`, `lot_code`, `adjustment_lb`, `reason` | **required**, must already exist (404 otherwise; never creates) |
| POST /void/{transaction_id} | ledger_corrections (append-only); sales_order_allocations (+ sales_order_allocation_reactivations on ship voids) | correction/void | floor | `reason` | n/a |
| POST /records/transactions/{id}/corrections | ledger_corrections; allocation side-writes on ship void/restore | correction — amend/void/restore | neither | `reason`; amend needs non-empty whitelisted `replacement_values` | n/a |
| POST /records/certifications | certifications | other — daily certification | neither | `business_date`, `certified_at` | n/a |
| POST /records/certifications/{id}/corrections | certifications (superseding row) | correction | neither | `certified_at`, `reason` | n/a |
| POST /products/quick-create | products; product_verification_history (best-effort) | master-data | neither | `product_name`, `product_type` | n/a |
| POST /products/quick-create-batch | products (type='batch' hardcoded) | master-data | neither | `product_name`, `category`, `production_context` | n/a |
| POST /lots/{lot_id}/reassign | lots (product_id), ledger_corrections (amend per posted line), lot_reassignments (best-effort) | correction — lot→product reassignment | neither | `to_product_id`, `reason_code` | n/a (lot by id in path) |
| POST /inventory/found | lots (find-or-create), transactions (adjust), transaction_lines, inventory_adjustments (best-effort) | adjustment — found inventory | neither | `product_id`, `quantity`, `reason_code` | **optional** — auto `YY-MM-DD-FOUND-NNN` |
| POST /inventory/found-with-new-product | products, lots, transactions (adjust), transaction_lines | adjustment + master-data | neither | `product_name`, `product_type`, `quantity`, `reason_code` | **optional** — same auto-gen |
| POST /products/{product_id}/verify | products; product_verification_history (best-effort) | master-data — verification | neither | `action` (verify/reject/archive) | n/a |
| POST /customers | customers | master-data | office | `name` | n/a |
| PATCH /customers/{customer_id} | customers; customer_aliases (clear-and-replace) | master-data | office | ≥1 field | n/a |
| POST /sales/orders | sales_orders, sales_order_lines; customers (auto-create) | SO | office | `customer_name`, `lines[]` (`product_name` + `quantity_lb` or `quantity`+`unit`) | n/a |
| POST /sales-orders/{so_number}/ready `[D]` | sales_order_flags (upsert) | SO — factory-ready flag | neither | `ready` (bool) | n/a |
| POST /sales/orders/{order_id}/allocations `[D]` | sales_order_allocations (upsert + expiry side-writes) | SO — allocation | neither | `line_id`; manual mode requires `quantity_lb` | lot **id** optional (manual mode); no lot code accepted |
| POST /sales/orders/{order_id}/allocations/{allocation_id}/release `[D]` | sales_order_allocations | SO — allocation release | neither | path params only | n/a |
| PATCH /lots/{lot_id}/received-at `[D]` | lots (received_at) | correction — backfill | neither | `received_at` (raw dict; tz-aware ISO-8601, not future) | n/a (lot by id) |
| PATCH /sales/orders/{order_id}/status `[D]` | sales_orders; sales_order_allocations on →cancelled | SO — status transition | both | `status` (state machine; shipped/partial_ship auto-only) | n/a |
| PATCH /sales/orders/{order_id} `[D]` | sales_orders (header) | SO — header edit | office | ≥1 of requested_ship_date/notes/notes_es/customer_id | n/a |
| POST /sales/orders/{order_id}/lines | sales_order_lines | SO — add lines | office | `lines[]` | n/a |
| PATCH /sales/orders/{order_id}/lines/{line_id}/cancel | sales_order_lines; sales_order_allocations (release) | SO — line cancel | office | path params only | n/a |
| PATCH /sales/orders/{order_id}/lines/{line_id}/update `[D]` | sales_order_lines; sales_order_allocations (shrink/split) | SO — line edit | office | query params `quantity_lb`/`unit_price` (≥1) | n/a |
| POST /sales/orders/{order_id}/ship (+ /ship/preview, /ship/commit `[D]`; commit also in floor schema) | shipments, transactions (ship per line), transaction_lines, sales_order_shipments, shipment_lines; sales_order_lines, sales_orders, sales_order_allocations | ship (SO fulfillment) | both | body optional (default preview); commit: `mode:"commit"` + `ship_all` or `lines[{line_id, quantity_lb}]` | n/a — lots auto-picked (allocation pins, then FIFO); no lot code accepted |
| POST /dashboard/api/notes `[D]` | notes | other — dashboard notes | neither | `category`, `title` | n/a |
| PUT /dashboard/api/notes/{note_id} `[D]` | notes | other | neither | ≥1 field | n/a |
| DELETE /dashboard/api/notes/{note_id} `[D]` | notes (hard delete) | other | neither | path param only | n/a |
| PUT /dashboard/api/notes/{note_id}/toggle `[D]` | notes | other | neither | path param only | n/a |
| PUT /admin/products/{product_id} | products | master-data | neither | ≥0 fields (no-op if none) | n/a |
| POST /admin/bom/{product_id}/lines | batch_formulas | master-data — BOM | neither | `ingredient_product_id`, `quantity_lb` | n/a |
| PUT /admin/bom/lines/{line_id} | batch_formulas | master-data — BOM | neither | all optional | n/a |
| DELETE /admin/bom/lines/{line_id} | batch_formulas | master-data — BOM | neither | path param only | n/a |
| POST /admin/product-bom | product_bom | master-data — FG→component map | neither | `finished_product_id`, `component_product_id` | n/a |
| DELETE /admin/product-bom/{mapping_id} | product_bom | master-data | neither | path param only | n/a |
| POST /admin/lots/merge | lots (source→merged), ledger_corrections (amend lines), ingredient_lot_consumption (in-place), sales_order_allocations (coalesce/repoint) | correction — duplicate-lot merge | neither | `source_lot_id`, `target_lot_id`, `reason` | lot **ids** required; codes resolved from DB |
| POST /schedule (action=confirm) | production_schedule (upsert); suggest/current are read-only | other — production scheduling | neither | `action`; confirm: `runs[]` each with `product_id`, `date`, valid `line_code` | n/a |

### Notes

- **Preview is the default.** ReceiveRequest/ShipRequest/MakeRequest/PackRequest/AdjustRequest and SO-ship all default `mode="preview"` (read-only); only `mode="commit"` writes. The `/preview` and `/commit` alias routes (main.py ~11021+) force the mode and are `include_in_schema=False` except `POST /sales/orders/{order_id}/ship/commit` (operationId `commitShipOrder`, exposed to the floor GPT).
- **occurred_at/backfill (PR #18, merged 2026-08-25):** the five inventory writes plus found-inventory and SO-ship now accept optional `occurred_at` (rejected if >5 min future) and `backfill: bool` (required for `occurred_at` >14 days old); stored on `transactions` with `created_at_source`.
- **Append-only ledger (migration 039):** voids/amends/restores never mutate `transactions`/`transaction_lines` — they append to `ledger_corrections`; reads resolve via `effective_status`.
- **Hidden customer creation:** `POST /ship` and `POST /sales/orders` auto-create `customers` rows via `resolve_customer_id(auto_create=True)` when no name/alias/fuzzy match (ambiguity 409s first).
- **Allocation side-writes:** ship/pack commits, voids/restores, SO status/line changes all pass through `_expire_auto_fifo_allocations` / `_shrink_overallocated_products`, which UPDATE `sales_order_allocations` beyond the obvious target.
- **Best-effort audit inserts** (failures swallowed, primary write still commits): `product_verification_history`, `lot_reassignments`, `inventory_adjustments`.
- **/adjust is the only core transaction requiring an existing lot code.** /receive, /make, /pack, /inventory/found* auto-generate codes when omitted; both ship paths never accept one.
- **Auth:** every write requires `X-API-Key` (master key) via `verify_api_key`; rows marked `[D]` additionally accept the scoped `DASHBOARD_API_KEY`. `/dashboard/api/*` GET endpoints are public, but the notes writes above are keyed.

## Tables with no write endpoint (dashboard-only reads, SQL-only, or migration-seeded)

| Table | How it gets data |
|---|---|
| adjustment_reason_codes | migration-seeded reference data (SQL only) |
| allergens | SQL only |
| product_allergens | SQL only |
| bom_lines / boms | legacy BOM pair — admin BOM endpoints write `batch_formulas` instead; SQL only |
| line_capacity_modes | scheduling config, SQL/migration only |
| production_lines | SQL/migration only |
| product_line_assignments | SQL/migration only |
| scheduling_config | SQL/migration only |
| reassignment_reason_codes | migration-seeded reference data |
| oauth_tokens | no app writes in main.py (unused/SQL only) |
| _backup_20260305_* (9 tables) | frozen 2026-03-05 snapshot set |

Every other table in the schema is reachable through at least one endpoint above. `production_schedule` is written only by `POST /schedule action=confirm`; `certifications` only by the two `/records/certifications` endpoints; `notes` only by the dashboard-keyed notes CRUD.

---

## Risk notes

Read-only follow-up (2026-08-25), same commit `65bca82`. Three targeted checks: lot-code generation, preview observability, best-effort audit inserts.

### 1. Lot-code auto-generation

Four generators, all inline except the receive one. **None of them include product, production line, or shift** — the code carries a date, a source tag, and a per-day sequence, nothing else.

| Path | main.py | Format string | Example | Sequence probe |
|---|---|---|---|---|
| `/receive` (preview L4116, commit L4169) | `generate_lot_code()` L4058–4088 | `f"{now:%y-%m-%d}-{shipper_code}-{seq:03d}"` | `26-08-25-COST-001` | `lots WHERE lot_code LIKE '26-08-25-COST-%' ORDER BY lot_code DESC LIMIT 1` |
| `/make` (preview L5532, commit L5613) | inline, duplicated | `f"B{now:%y-%m%d}-{seq:03d}"` | `B26-0825-001` | `LIKE 'B26-0825-%'` |
| `/inventory/found` (L8002) and `/inventory/found-with-new-product` (L8107) | inline, duplicated | `f"{now:%y-%m-%d}-FOUND-{seq:03d}"` | `26-08-25-FOUND-001` | `LIKE '26-08-25-FOUND-%'` |
| `/pack` (preview L5976, commit L6103) | — | **no generation**: `req.target_lot_code` → else the first allocated source lot's `lot_code` → else literal `"UNKNOWN"` (preview only) | inherits e.g. `B26-0825-001` | n/a |

`shipper_code` = `shipper_code_override.upper()[:4]` if given, else the first 4 **alpha** characters of `shipper_name` uppercased, else `"UNKN"`.

**Can two different products get identical codes on the same date? Yes — three distinct ways.**

1. **`/pack`, by design.** The packed SKU's lot reuses the source bulk lot's code verbatim (L6103–6106), so one code string legitimately spans the bulk product and every finished SKU packed from it. `lots` is unique on `(product_id, lot_code)`, not on `lot_code`, so this is permitted. **A lot code alone is not a key** — every lookup must carry `product_id` (`GET /lots/{lot_code}` already takes an optional `product_id`; without it, a shared code is ambiguous).
2. **`/make`, by race.** `/receive` takes `pg_advisory_xact_lock(1)` (L4162) and both found paths take `pg_advisory_xact_lock(2)` (L7994, L8099) before generating, which serialises those generators. **`/make` takes no advisory lock.** Two concurrent make commits read the same `MAX` and mint the same `B{yy-mmdd}-{seq}`; for two different products both inserts succeed → identical codes. For the *same* product it is worse: `find_or_create_lot()` uses `INSERT … ON CONFLICT (product_id, lot_code) DO NOTHING` (L4037), so the second batch is silently folded into the first batch's lot — two production runs, one lot, no error, `lot_is_new: false` the only signal in the response.
3. **Sequence reset from a manual code sharing the prefix.** The probe parses `int(lot_code.split('-')[-1])` and falls back to `seq = 1` on `ValueError`. A hand-entered code that matches the prefix but ends non-numerically (e.g. `26-08-25-COST-A`) sorts *above* the numeric ones in `ORDER BY lot_code DESC` (ASCII `'A'` > `'9'`), so the parse fails and the counter restarts at `001` — colliding with the existing `-001`. Same silent-merge consequence as above. The `:03d` padding also breaks lexical ordering past 999 lots on one prefix (`'1000' < '999'`), where the counter would stick.

Two further notes:

- **The date in the code is the entry date, never the event date.** All four generators call `get_plant_now()`. PR #18's `occurred_at` / `backfill` only reach `transactions`; a receive backfilled to 2026-08-01 still mints a `26-08-25-…` lot code. Lot codes cannot be used to infer when the material actually arrived or was produced.
- **`shipper_code_override` is not sanitised** (`.upper()[:4]` only, L4062), so an override containing `%` or `_` becomes an unintended `LIKE` wildcard in the sequence probe. Distinct suppliers can also collapse to one 4-letter code (they then share a sequence — no duplicate, but ambiguous provenance).

### 2. `mode="preview"` — persistence and observability

Verified by walking each preview branch and every helper reachable from it (`/receive` L4103–4157, `/ship` L5111–5224, `/make` L5430–5584, `/pack` L5919–6019, `/adjust` L6227–6268, plus SO-ship L10706–10786).

**Nothing is written and nothing is logged on the success path.**

- No `INSERT`/`UPDATE`/`DELETE` in any preview branch, directly or through helpers. The two write-capable helpers reachable from a preview are both gated off: `resolve_customer_id()` is called with `auto_create=False` (L5117), and `_expire_auto_fifo_allocations()` (L394) only fires when `available_lots_for_product(persist_expired=True)`, which additionally requires `lock=True` — preview call sites pass neither (L5131, L5931) and the SO-ship preview passes `lock=False, persist_expired=False` explicitly (L10744–10750).
- The only `logger` call in any preview branch is `logger.error("… preview failed")` on the 500 path. Successful previews emit no application log line.
- No request-logging middleware exists — only `CORSMiddleware` and `write_response_envelope` (L108), neither of which touches the DB.
- Previews issue no `nextval`, so they leave no sequence gap either.

**Detecting an abandoned preview: not possible from the database.** A preview that is never committed is indistinguishable from a request that never happened. The only residue is the HTTP access log (Railway/uvicorn, retention-limited, method + path only, no body):

- Callers using the alias routes (`POST /receive/preview` etc., L11021+) are distinguishable *by path*.
- Callers using the base route with `mode` in the body — which is the default, `mode="preview"` — are **not**; `POST /receive` looks identical for preview and commit.

Corollary for forensics: gaps in `transactions.id` indicate *attempted commits that rolled back*, not previews.

Related drift risk: the lot code shown in a preview is recomputed at commit (L4169, L5613). Any intervening receive/make on the same prefix shifts the sequence, so the operator can be shown `B26-0825-003` and get `B26-0825-004`. The preview reserves no inventory and takes no lock, so its allocation plan and sufficiency check can also be stale by commit time.

### 3. Best-effort audit inserts — failure behaviour

| Table | Site | On failure | Visible in response? |
|---|---|---|---|
| `product_verification_history` | `/products/quick-create` L7767–7773; `/products/{id}/verify` L8222–8228 | `except Exception: pass` — **not even logged** | No |
| `lot_reassignments` | `/lots/{lot_id}/reassign` L7931–7945 | `logger.warning("Failed to record lot reassignment history: …")` | Indirectly — `reassignment_id` comes back `null` while `success: true` |
| `inventory_adjustments` | `/inventory/found` L8040–8050 | `logger.warning("Failed to record inventory adjustment: …")` | No |

**The swallow is not safe.** Each of these runs on the *same* psycopg2 connection and transaction as the primary write. A DB-level error (constraint, `NOT NULL`, `value too long`) aborts the whole transaction; every later statement in the block would fail, and `get_db_connection()` (L1976–1990) then calls `conn.commit()` on the aborted transaction — Postgres turns that `COMMIT` into a `ROLLBACK` and psycopg2 raises nothing. **The primary write is silently discarded while the endpoint returns `success: true` with a `lot_id`/`product_id`/`transaction_id` that does not exist.** In all three cases the audit insert is the last statement before `return`, so nothing downstream surfaces the abort.

Confirmed empirically against the local test DB (`factory_ledger_test`, psycopg2 2.9.9): primary insert → deliberately failing second insert caught and swallowed → `conn.commit()` raised nothing → surviving rows: **0**.

Reachable failure vectors, given none of the request models bound string length:

- `inventory_adjustments.found_location varchar(200)`, `estimated_age varchar(50)` (default `"unknown"`), `suspected_supplier varchar(200)`, `adjusted_by varchar(100)`, `reason_code varchar(50) NOT NULL`, `uom varchar(20) NOT NULL` — `AddFoundInventoryRequest` (L2580) declares all of these as unbounded `str`/`Optional[str]`. A long free-text `found_location` from a GPT caller is enough to roll back a found-inventory posting that reports success.
- `lot_reassignments.reason_code varchar(50) NOT NULL`, `reassigned_by varchar(100)`, `from_product_name`/`to_product_name varchar(200) NOT NULL`.
- `product_verification_history.to_status varchar(20) NOT NULL`, `action varchar(50) NOT NULL`, `performed_by varchar(100)`.

Distinction worth keeping: a *Python-level* failure inside the `try` (e.g. `float(None)` on `quantity_on_hand` before `execute`) is harmless — no SQL ran, the transaction is intact. Only a failed `cur.execute` poisons it, and that is exactly the case the handlers hide.

Two coverage gaps found alongside:

- `/inventory/found-with-new-product` (L8072–8152) writes `products`, `lots`, `transactions`, `transaction_lines` but **never inserts into `inventory_adjustments`** — found inventory created through that path has no adjustment audit row at all, so the table under-counts found events.
- `product_verification_history` is written on quick-create and verify, but the `except: pass` means a status change can land in `products` with no history row and no trace anywhere — no log line, no response field.

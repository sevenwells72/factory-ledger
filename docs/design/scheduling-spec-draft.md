# Production scheduling — discovery and spec draft

Date: 2026-09-14. Branch: `docs/scheduling-spec` (off `origin/main` @ `72b5546`).
Status: **DRAFT — discovery only. No code, no migration, no schema change in this PR.**

Line numbers below are taken from the checkout at `87e95ef`
(`feat/fr15-user-attribution`, PR #49, which is `main` @ `72b5546` plus the
actor-attribution commits). PR #49 is unmerged; every citation into the actor
mechanism (`_authorize_api_key`, `caller_source_tag`, `actors`) is on that branch
only. Everything else is identical on `main`. Line numbers drift; grep by
symbol before editing.

Owner decisions this spec is built on, verbatim from the brief:

1. Scheduling coverage **replaces allocations as the Health signal**. Allocations
   stay as data but stop driving alarms.
2. Goal is daily factory use as fast as possible: prefer the **smallest model**
   that lets Health say "short and nobody is making it" vs "short but scheduled".
3. The production schedule currently lives **only in WhatsApp**. That is the gap.

---

## Part 1 — what exists

### 1a. How production is recorded today

**There is no `batches` table and no `inventory_events` table.** The ledger is
`transactions` + `transaction_lines`, read through the void/amend layer.

| Object | Where | What it is |
|---|---|---|
| `transactions` | `tests/schema/schema.sql:976-999` | One row per event. `type` is **unconstrained text** (no CHECK, no enum); de-facto values `receive, ship, make, pack, adjust`. Carries `timestamp` (plant-local, from `get_plant_now()` `main.py:3172`), `occurred_at timestamptz NOT NULL`, `business_date date NOT NULL` (trigger-derived in `America/New_York`, `migrations/046_inventory_occurred_at.sql:31-33`), `operator_id`, `expected_receipt_id`. |
| `transaction_lines` | `schema.sql:928-936` | `(transaction_id, product_id, lot_id, quantity_lb)`. Positive = into the lot, negative = out. |
| `ledger_corrections` | `schema.sql:905-920` | `event_type CHECK IN ('amend','void','restore')` (`:915`). A void appends a row and posts **no reversal lines** (`main.py:9227-9233`). |
| `ledger_current_transactions` / `ledger_current_transaction_lines` | `schema.sql:1005-1063`, `:943-975` | The views that compute `effective_status`. |
| `POSTED_LINES` | `main.py:380-385` | The one SQL fragment every balance must read through: `effective_status = 'posted'` only. Helpers `lot_on_hand()` `:387`, `fifo_lot_balances()` `:408`, `_product_on_hand()` `:571`. |
| `ingredient_lot_consumption` | `schema.sql:809-817` | Traceability mirror of every negative make/pack input line, stored **positive**. |
| `lots` | `schema.sql:1197-1216` | `entry_source` for production lots is `'production_output'` (make) or `'pack_output'` (pack) — `main.py:9302-9304`. |
| `products` | `schema.sql:1350-1399` | `type CHECK IN ('ingredient','packaging','batch','finished','consumable')` (`:1397`). Production-relevant: `default_batch_lb`, `yield_multiplier`, `case_size_lb`, `parent_batch_product_id`, `no_production` (migration 038), `pack_format CHECK IN ('10lb','25lb','bagged')` (migration 040), `units_per_case`, `bags_per_case`, `low_stock_threshold` (043). |
| `trace_events` / `trace_event_lots` | `migrations/048_trace_tables.sql:35-64`, `:85-100` | `event_type CHECK IN ('receive','make','pack','ship','adjust','void','restore','amend','merge')` (`048:39-40`) — the only real event-type enum in the schema. |

**Bake line / pack line as product attributes: not found.** `products.bake_line`
was added on 2026-05-18 and dropped the next day
(`archive/migrations-applied/sql/034a_fruit_nut_bom_bake_line_copack.sql:53`,
`036_drop_bake_line.sql:36`). Line routing lives in `product_line_assignments`
(§1d), not on `products`.

**Endpoints that write a production event** (decorator lines):

| Route | Line | Writes |
|---|---|---|
| `POST /make` | `main.py:8070` | `transactions` type `'make'` (`:8292-8299`), one positive output line (`:8304-8307`, `total_output = default_batch_lb × batches × yield_multiplier` `:8235-8237`), one negative line + one ILC row per ingredient lot consumed FIFO (`:8353-8360`, `:8376-8377`), a `production_output` lot with code `B{yy-MMDD}-{seq}` (`:8256-8261`), trace event (`:8388-8394`). Ingredient need is `batch_formulas.quantity_lb × batches` (`:8272-8278`, `:8322`) — **scaled by batches, not by output pounds**. Serialized by `pg_advisory_xact_lock(3)` (`:8252`). |
| `POST /pack` | `main.py:8577` | `transactions` type `'pack'` (`:8778-8785`), `total_lb = cases × case_weight` (`:8690-8696`), one positive FG line (`:8791`), negative + ILC per source batch lot (`:8795-8796`), formula add-ins (`:8470-8576`, `:8843-8844`), trace (`:8856-8861`), `pack_output` lot (`:8886`). **No BOM and no packaging consumption** on the batch→FG hop. |
| `POST /make/preview|commit`, `POST /pack/preview|commit` | `main.py:15026-15041` | Aliases forcing `mode`. |
| `POST /adjust` | `main.py:8907` | type `'adjust'`, one signed line. |
| `POST /inventory/found`, `/found-with-new-product` | `main.py:10696`, `:10800` | type `'adjust'` + `inventory_adjustments` row. |

Read-side production surfaces already on the dashboard allowlist:
`GET /production/requirements` (`main.py:17323`, BOM explosion for one product),
`GET /production/today-tile` (`:17510`, posted make/pack for a plant day),
`GET /production/day-summary` (`:17635`, per-lot make/pack/adjust for a day).
`GET /dashboard/production` (`:4338`) reads the `production_history` view, which
reads **raw** `transactions` (`schema.sql:3809-3824`) and therefore still shows
voided makes — do not build on it.

**What a completed make looks like in the ledger** (N batches of batch product P,
k ingredient-lot slices), confirmed by `tests/test_trace_emission.py:286-299`:

```
transactions               1 row   type='make', occurred_at, business_date, status='posted'
lots                       0–1 row entry_source='production_output'
transaction_lines          1 row   (+total_output_lb, output lot)
transaction_lines          k rows  (−take, ingredient lot)      FIFO slices
ingredient_lot_consumption k rows  (+take, ingredient lot)      mirror
trace_events               1 row   event_type='make', epcis_type='transformation'
trace_event_lots           k+1     role='input' (−) ×k, role='output' (+) ×1
```

A completed pack is the same shape with type `'pack'`, the FG as output, source
batch lots (and any add-in lots) as inputs. Nothing else is written: no
`production_schedule` change, no plan/actual link
(`FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:720-730`, gap #16 at `:1592`).

### 1b. How Health v2.1 computes shortage per SO line

One query, no writes, no locks: `SALES_ORDER_READINESS_SQL` (`main.py:11850-11987`),
executed once per page in `_load_sales_order_readiness()` (`:12108`, `:12113`).
Per-line arithmetic in `_line_readiness()` (`:12009`):

```
remaining        = max(0, ordered − shipped_effective)            :12014
available        = on_hand − allocated_product                     :12021  (informational)
coverable        = max(0, on_hand − allocated_others)              :12022
shortage         = max(0, remaining − coverable)                   :12023
unallocated_need = max(0, remaining − allocated)                   :12024
```

`on_hand` is the SKU-level posted balance (`on_hand_sku` CTE `:11891-11895`);
`allocated_*` come from the `live_alloc` CTE (`:11907-11914`, `status='active'`
and not expired). So **another order's reservation still counts against this
order's shortage today** (`allocated_others`), while this order's own reservation
does not. `inbound_open_lb` (`:11943-11949`) sums open `expected_receipts` but is
never added to availability; it only produces the `warn` blocker `inbound_cover`
(`:12081-12082`).

Line-level `inventory_ready` (`:12084-12087`) **is** allocation-gated:
`remaining ≤ ε OR (allocated ≥ remaining AND shortage ≤ ε)`. That feeds
`dispatch_ready` (`:12175-12177`), `GET /sales/orders/fulfillment-check`
(`:12834`) and the dashboard readiness tile — not Health.

Tiering is `compute_so_health()` (`main.py:12402`), fed only by the readiness
dicts, no SQL of its own, `today = _factory_today()` (`:12313`). Constants:
`SO_HEALTH_CRITICAL_DAYS_DEFAULT = 5` (`:12279`), `SO_HEALTH_WARNING_DAYS_DEFAULT = 10`
(`:12284`), `SO_HEALTH_READY_DAYS = 2` (`:12288`), `FACTORY_TZ_DEFAULT` (`:12274`),
env read per call via `_so_health_window_days()` (`:12318`). The rule table is
documented at `docs/design/so-state-model-findings.md:893-908`; the code path is
`:12467-12530` (shortage → `speaks_up` `:12485`, `critical` `:12491`; overdue-with-stock;
Not Ready to Ship) and `level` at `:12576`. Tiers are `critical | warning | quiet`;
`info` is a list, never a level.

Callers: `_so_derived_fields()` (`:12581`) from `GET /sales/orders` (`:12602`, call at
`:12705`) and `GET /sales/orders/{order_id}` (`:13556`, call at `:13597`).
`GET /sales/orders/counts` (`:13012`) shares the clock but emits no health.

**Every place "allocations not enforced" is emitted or consumed:**

| Where | Line | Text |
|---|---|---|
| API, the only producer | `main.py:12571` | `note = "" if enforced else " (allocations not enforced)"` — appended to `f"{n} lb not allocated{where}{note}"` (`:12572-12574`), once per order, `info` list only. `enforced = _allocations_enforced()` (`:438-447`, env `ALLOCATIONS_ENFORCED`, default false). |
| API, the rows behind it | `main.py:12547-12563` | `info_detail = [{line_id, sku, product_name, unallocated_lb}]` |
| Dashboard list | `dashboard/so-list.js:28` | regex strip of `(allocations not enforced)` when `omitAllocationNote`; `:30` "Show allocation details" popover from `info_detail`. |
| Dashboard detail | `dashboard/dashboard.js:3230` | detects the phrase in reasons/info; `:3251` renders `Allocations not enforced: reservations do not prevent shipping.`; `:3266-3267` per-line "Allocation details"; `:3281` Allocated/Unallocated columns. |
| Tests | `tests/test_sales_order_state_model.py:1189-1210`, `:1366-1368` | pin the exact string, the once-per-order count, and the on/off pair. |
| Write-path warnings (not Health) | `main.py:799-848` | "Allocation enforcement is on/off…" on ship/pack previews. Untouched by this spec. |

Related blockers that carry allocation vocabulary in `readiness.blockers`:
`unallocated` (`:12067-12068`) and `partial_allocation` (`:12069-12071`).

### 1c. Expected Receipts and Supplies

**Expected receipts** — `migrations/041_expected_receipts.sql:79-92`
(`product_id, supplier_id, expected_qty lb, expected_date date, status CHECK IN
('open','closed','cancelled'), created_by, updated_at`; `source_document_id`
from 049). Design invariant, verbatim (`041:14-18`): *"there is NO stored
remaining/received balance anywhere… Nothing here is read by any on-hand /
availability query."* Remaining is `expected_qty − SUM(posted linked receipt lines)`
via `EXPECTED_RECEIPT_RECEIVED_SQL` (`main.py:5613`). There is no
"receive this ER" endpoint: `POST /receive` (`:5298`) FIFO-matches an open ER by
product + supplier (`:5399-5406`, rule `:5691-5697`), writes the link on the
transaction header (`:5408-5418`) and auto-closes via `settle_expected_receipt()`
(`:5715`). Overdue is derived (`_serialize_expected_receipt` `:5649`, `:5656`).
Endpoints: `:5851 POST`, `:5891 GET`, `:5937 GET one`, `:5974 PATCH`, intake `:7005/:7068/:7076`.
There is **no ETA beyond `expected_date`** and no lead-time field anywhere.

**"Available" today** means the posted-only ledger sum, nothing else, except on the
sales-order path where `available_lots_for_product()` (`:612`) subtracts active
reservations. No function anywhere projects on-hand + expected receipts
(grep `projected|forecast` → 0 hits). Three legacy views still read raw
`transaction_lines` and bypass the void layer: `inventory_summary`, `lot_balances`,
`low_stock_alerts` (`schema.sql:3729-3773`), served by `GET /dashboard/inventory`
(`main.py:4298-4302`) and `/dashboard/low-stock` (`:4308`).

**Supplies** — not a table. `migrations/043_supplies.sql` widens
`products.type` to admit `'consumable'` (`043:28-32`), adds
`products.low_stock_threshold` (`:35-44`) and a `supply_requests` queue
(`:47-68`, "we need X", no balance). `PRODUCT_CATEGORIES = ("ingredient","packaging","consumable")`
(`main.py:7389`); endpoints `:7433`, `:7475`, `:7565`, `:7609`, `:7641`.
Packaging exists as 32 `type='packaging'` products, but **nothing consumes
packaging**: `/pack` deducts no packaging product, and the packaging rows that
exist in `product_bom` for some SKUs (e.g. 31012 → bag 21003, box 21011,
`FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:498`) are never read for consumption
(`:487`, `:505-511`).

**BOM.** The live recipe is `batch_formulas` (`schema.sql:522-529`): batch
product → ingredient, `quantity_lb` **per single batch of `default_batch_lb`**,
`exclude_from_inventory` for water etc. No version, no effective date. The
FG → batch hop is `product_bom` (`schema.sql:1501-1509`, `finished_product_id,
component_product_id, quantity, uom`) **and/or** `products.parent_batch_product_id`
— two mechanisms, neither complete (`SYSTEM_KNOWLEDGE.md:452`). `boms` / `bom_lines`
(`schema.sql:590-602`, `:555-563`) are dead: zero rows, no reader.

Is ingredient/packaging consumption derivable from BOM × planned quantity?

| Hop | Derivable now? | Evidence |
|---|---|---|
| batch product → ingredients | **Yes.** All 25 active batch products have formulas and a `default_batch_lb`. Three implementations exist: `/production/requirements` (`main.py:17404`), scheduler `_explode_ingredients()` (`:18187-18217`), `/make` preview (`:8097`). | `SYSTEM_KNOWLEDGE.md:517` |
| finished SKU → batch product | **Partially.** 34 active finished products have no `product_bom` row; `/production/requirements` 404s on them (`:17359-17360`). Some have only `parent_batch_product_id` (the SS Classic #9 family, `:1390`). `product_bom.quantity` is ignored by the scheduler (`SYSTEM_KNOWLEDGE.md:1391`). | `SYSTEM_KNOWLEDGE.md:516`, `:452` |
| anything → packaging | **No.** No per-case packaging quantity exists; packaging rows in `product_bom` are unread. | `:487`, `:505-511` |

One known mass inconsistency to expect in the numbers: 90008 Granola Fruit Nut
Batch has formula rows totalling 25 lb against `default_batch_lb = 384.52`
(`SYSTEM_KNOWLEDGE.md:517`).

### 1d. Existing planning artifacts — there are several

The brief says the schedule lives only in WhatsApp. That is true of the *live*
schedule; the repo nonetheless carries a full planning stack, all of it dormant:

1. **Migration 004 scheduling tables**, live in prod
   (`archive/migrations-applied/sql/004_production_scheduling.sql`):
   `production_lines` (4 rows: `granola`, `coconut`, `bulk_pack`, `pouch`;
   `schema.sql:1619-1624`), `line_capacity_modes` (`:1064`; live values at
   `SYSTEM_KNOWLEDGE.md:1373-1382`), `product_line_assignments` (`:1535-1539`;
   **63 live rows**: 18 batch → granola, 4 batch → coconut, 32 finished → bulk_pack,
   9 finished → pouch, `SYSTEM_KNOWLEDGE.md:379`), `scheduling_config` (`:1917`), and
   **`production_schedule`** (`:1651-1668`):
   ```
   id, schedule_date date NOT NULL, line_id int NOT NULL FK, product_id int NOT NULL,
   planned_batches, planned_quantity_lb, planned_bags, workers_assigned int NOT NULL,
   status CHECK IN ('planned','confirmed','in_progress','completed','cancelled'),
   linked_order_numbers text[], overproduction_lb, overproduction_reason, notes,
   created_at, confirmed_at, created_at_source
   UNIQUE (schedule_date, line_id, product_id)          schema.sql:2945-2946
   ```
   **Zero rows in production** at every audit (`SYSTEM_KNOWLEDGE.md:1384`,
   `docs/data-health-baseline-2026-08-24.md:702`, `week1-scoring-extract-2026-08-13.md:213`).
   No attribution columns, no `updated_at`, no link to a ledger transaction; nothing
   ever writes `in_progress` or `completed`.
2. **`POST /schedule`** (`main.py:18642`), master-key only, three actions:
   `suggest` (`_handle_schedule_suggest` `:18390`) — the 7-day tactical engine
   (`:17851-18470`: calendar `:17855`, line config `:17880`, product→line map `:17910`,
   demand `:17920`, inventories `:17937`/`:17951`, BOM `:17965`, simulated allocation
   `:18020`, ingredient explosion `:18187`, day packing `:18220`); `confirm`
   (`:18506`, upsert into `production_schedule` `:18536-18559`); `current` (`:18575`).
   Not in either GPT schema; explicitly denied to the dashboard key
   (`tests/test_dashboard_api_key.py:88`, `:137`). Documented limitations in
   `SYSTEM_KNOWLEDGE.md` gaps 13–16 (`:1580-1592`): ignores `no_production`, cannot
   route Graham 31012 or the SS Classic #9 family, treats pack lines as one
   batch/day, and never reconciles to actuals. Its demand filter still uses legacy
   `so.status` (`:17929`), not the authoritative `state` from migration 051.
3. **`dashboard/scheduler/seven-wells-production-board.html`** (1,666 lines): a
   self-contained browser planner with its own embedded catalog (`CATALOG` at
   `:289`), 84-day greedy algorithm, `localStorage` key `sw_prodboard_v1`
   (`:395-397`). **It neither reads nor writes any Factory Ledger API**
   (`SYSTEM_KNOWLEDGE.md:945`). Changelog rows 42–47 record its phases; rows 42–46
   were never deployed and live on remote branches `feat/pan-product-mapping`
   (`bd7bade`), `feat/day-schedule-panel` (`8229f85`), `feat/schedule-text-v2`.
4. Historical: `demand_plan.html`, `planner_v2.html`, `render_planner_v2.py`,
   `render_demand_plan.py` — absent from every current branch (`STATUS.md:124-125`).
5. `products.no_production` (migration 038) exists specifically to keep 26 resale
   SKUs out of "the production scheduler" (`038:1`).

`cns_planner` and `five-tier`: **not found** anywhere in the repo.

### 1e. Floor GPT surface and the allowlist

`openapi-gpt-v3.yaml` exposes **exactly 30 operations** (30 `operationId`s, 30
method keys; `gpt-configs/schemas/openapi-floor.yaml` has 22). The hard rule in
`CLAUDE.md` is that this never exceeds 30. **No endpoint in this spec is added to
either yaml.**

After PR #49 the allowlist is unchanged in location and shape:
`DASHBOARD_KEY_ALLOWLIST = frozenset({...})` at **`main.py:2668-2762`**, keyed on
`(METHOD, route template)` via `_route_key()` (`:2765`). It is consulted by the
single choke point `_authorize_api_key()` (`:3069-3100`): master key → everywhere;
`DASHBOARD_API_KEY` → allowlist only; **actor key (new in #49) → the same
allowlist, with `request.state.actor` attached** (`:3092-3099`). So one allowlist
entry admits both the shared dashboard key and every per-person key. Attribution
then comes for free through `caller_source_tag(request)` (`:5556`, actor name →
`'dashboard'` → body tag → NULL) — this is the "PR #49 actor mechanism" the new
tables should stamp. `tests/test_dashboard_api_key.py:81-90` pins that no allowlist
path starts with `/make`, `/pack`, `/adjust`, `/admin`, `/void` or equals `/schedule`;
`/production/runs…` passes that guard.

Precedent for allowlist-only routes: the existing block comment at `:2669-2672`
(`/auth/whoami`, "Deliberately NOT in openapi-gpt-v3.yaml") and the 051 exits at
`:2716-2722`.

---

## Part 2 — proposed design, smallest viable

### 2f. Schema

**Recommendation: two new tables, `production_runs` and `run_coverage`. Leave
`production_schedule` and `POST /schedule` untouched and dormant** (retire in a
later, separate PR). This is owner decision Q1 below. Reasons not to reuse
`production_schedule`:

* `UNIQUE (schedule_date, line_id, product_id)` forbids two runs of one product on
  one line on one day, which is a normal thing to plan (morning bake for order A,
  afternoon for order B); `_handle_schedule_confirm` depends on that constraint for
  its `ON CONFLICT` (`:18541`), so dropping it breaks the dormant engine.
* `line_id NOT NULL` and `workers_assigned NOT NULL` force data the floor will not
  have on day one.
* Its status vocabulary (`confirmed`, `completed`) and its `linked_order_numbers
  text[]` (order-level, by number, not line-level) do not match the coverage model.
* No attribution, no `updated_at`, no actual link — every column this spec needs is
  a migration anyway, and the table is empty, so reuse saves nothing.

```sql
production_runs
  id                    serial PRIMARY KEY
  product_id            integer NOT NULL REFERENCES products(id)
  planned_qty_lb        numeric(14,4) NOT NULL CHECK (planned_qty_lb > 0)
  planned_date          date NOT NULL
  line_id               integer REFERENCES production_lines(id)        -- nullable
  status                text NOT NULL DEFAULT 'planned'
                        CHECK (status IN ('planned','in_progress','done','cancelled'))
  notes                 text
  actual_transaction_id integer REFERENCES transactions(id)            -- nullable, see 2g
  created_at            timestamptz NOT NULL DEFAULT clock_timestamp()
  created_at_source     text NOT NULL DEFAULT 'database'
  created_by            text                                           -- caller_source_tag()
  updated_at            timestamptz NOT NULL DEFAULT clock_timestamp()
  updated_by            text                                           -- caller_source_tag()
  INDEX (planned_date), INDEX (product_id, status)
  TRIGGER trg_production_runs_created_at BEFORE INSERT OR UPDATE
          EXECUTE FUNCTION ledger_enforce_created_at()   -- same as production_schedule, schema.sql:3909

run_coverage
  run_id                integer NOT NULL REFERENCES production_runs(id)
  sales_order_line_id   integer NOT NULL REFERENCES sales_order_lines(id)
  qty_lb                numeric(14,4) NOT NULL CHECK (qty_lb > 0)
  created_at            timestamptz NOT NULL DEFAULT clock_timestamp()
  created_by            text
  PRIMARY KEY (run_id, sales_order_line_id)
  INDEX (sales_order_line_id)
```

`line_id`: **the data supports it** — 4 lines and 63 product assignments exist
(§1d). On create, default `line_id` from `product_line_assignments` when the
product has exactly one assignment; otherwise leave NULL. It is a label for the
board, not a capacity input.

`product_id` is the product the run yields. For coverage to be meaningful it must
equal `sales_order_lines.product_id`, i.e. the **finished SKU** (pack-out), not the
batch. A run of a batch product is allowed (the floor bakes batches) but cannot
carry coverage in v1; batch count goes in `notes`. This is owner decision Q2.

Application invariants, enforced under lock (§2k), not by constraint:

* `run_coverage.qty_lb` summed per run ≤ `planned_qty_lb`.
* `run.product_id = line.product_id`; `run.status ∉ {done, cancelled}` at set time;
  `line.line_status <> 'cancelled'`; order `state = 'open'`.
* Coverage on a run that later goes `done`/`cancelled`, or on an order that later
  exits, is **left in place** and filtered at read time. No exit path is modified.

**Deliberately NOT modeled, and why:**

| Not modeled | Why |
|---|---|
| Shifts, start times, sequence within a day | The WhatsApp schedule is "what, how much, which day". Health only needs the day. |
| Capacity, workers, batches/day | `line_capacity_modes` exists but its numbers are aspirational and the 7-day engine that used them was never adopted. A run that cannot fit is the owner's call, not a 409. |
| Bake → pack as two linked runs | Doubles the writes per order for no Health gain. One run = one finished SKU on one day. |
| Ingredient reservations / purchase suggestions | Materials is a derived read (§2i). Reserving would re-create the allocations problem one level down. |
| Plan-vs-actual variance history | `actual_transaction_id` is enough to answer "did it happen". Variance is a report, later. |
| Order-level coverage (`linked_order_numbers`) | Health is per line; coverage is per line. |

### 2g. Linking actuals — recommend manual, with a read-side hint

Options considered:

* **Automatic by product + date window** — a posted `make`/`pack` for the same
  product with `business_date` within ±1 day of `planned_date` flips the run to
  `done`. Rejected as the primary mechanism: two runs of one SKU in a week become
  ambiguous, partial packs (one run, two pack transactions) have no rule, and it
  makes a ledger write mutate a planning row — the codebase's stated invariant is
  that a schedule row is never evidence of production and vice versa
  (`SYSTEM_KNOWLEDGE.md:77`, `:720-730`). `/make` and `/pack` keep their lock
  rules out of the SO graph (`so-state-model-findings.md:503-512`); adding a
  `production_runs` write inside them is exactly the cross-graph edge the lock
  audit says to avoid.
* **Manual** — `PATCH /production/runs/{id}` with `status: done` and an optional
  `actual_transaction_id`. The API validates the transaction is posted, is
  `make`/`pack`, and has a positive output line for `run.product_id`.
* **Both** — manual write, automatic *hint*.

**Recommendation: both, in that shape.** The run list returns
`candidate_actuals: [{transaction_id, type, business_date, output_lb}]` for
`planned`/`in_progress` runs (posted make/pack of the same product, `business_date`
in `[planned_date − 1, planned_date + 1]`, not already linked to another run). The
board shows "Posted 412 lb on Sep 16 — mark done?" and the operator confirms with
one tap. The ledger never writes to `production_runs`. Actual quantity is derived
at read time from the linked transaction's posted output line through
`POSTED_LINES`, never stored — a void of the transaction makes `actual_qty_lb`
read as 0 on its own.

### 2h. Health v3 rules

Format mirrors `so-state-model-findings.md:893-931`. Evaluated on **open orders
only**; highest tier wins; every applicable reason is listed; `today` is
`_factory_today()`. The shape `{level, reasons, info, info_detail}` is unchanged.

New per-line readiness fields, computed in a `coverage` CTE added to
`SALES_ORDER_READINESS_SQL` and surfaced by `_line_readiness()`:

```
covered_lb        = SUM(run_coverage.qty_lb) over runs with status IN ('planned','in_progress')
uncovered_short_lb = max(0, shortage_lb − covered_lb)
covered_short_lb   = min(shortage_lb, covered_lb)
runs              = [{run_id, planned_date, status, qty_lb}]  ordered by planned_date
```

`shortage_lb` keeps its v2.1 formula (`:12023`). `done` runs do not count as
coverage: once the pack posts, `on_hand` rises and the shortage shrinks by itself;
a run marked done with nothing posted is a data error the board surfaces, not
something Health should paper over.

| Level | Condition | Example reason |
|---|---|---|
| `critical` | `uncovered_short_lb > 0` on any line **and** `ship_by ≤ today + SO_HEALTH_CRITICAL_DAYS` (past due included) | `Short 735 lb on 2 lines, not scheduled — ships in 3 days` |
| `warning` | `uncovered_short_lb > 0` with `ship_by ≤ today + SO_HEALTH_WARNING_DAYS` but outside the critical window, **or** no ship date | `Short 500 lb, not scheduled — ships in 8 days` |
| `warning` | **Late run**: any covering run (`planned`/`in_progress`) has `planned_date > ship_by` | `Run for 500 lb planned Sep 20 — after ship date Sep 18` |
| `warning` | Overdue (`ship_by < today`, fulfillment ≠ shipped) with **no** shortage at all | `12 days overdue — stock on hand` (unchanged) |
| `warning` | Ready to Ship not set, `ship_by ≤ today + 2`, and **nothing short** | `Not Ready to Ship — ships tomorrow` (unchanged) |
| `info` | `uncovered_short_lb > 0` with `ship_by` beyond the warning window | `Short 1,400 lb, not scheduled — ships in 13 days` |
| `info` | **Covered shortage**: `covered_short_lb > 0`, stated with the covering run date(s) | `500 lb scheduled Sep 16 — ships in 8 days` (`Sep 16–17` when several runs) |
| `info` | **Run overdue**: a covering run has `planned_date < today` and `status = 'planned'` | `Run for 500 lb planned Sep 12 has not started` |
| `quiet` | Nothing above applies, or the order is closed/cancelled | — |

Precise rules:

1. The uncovered-shortage reason is aggregated to one line per order exactly as
   v2.1 aggregates shortage (`:12473-12511`): `on N lines` when N > 1, the
   `_so_ship_phrase()` tail (`:12382`), wording identical whether it lands in
   `reasons` or `info`. The words `, not scheduled` are the only addition.
2. The covered-shortage sentence is `info` and never moves `level`. It is emitted
   even when an uncovered remainder exists on the same order, so an order that is
   half-covered reads as two facts: `Short 300 lb, not scheduled — ships in 3 days`
   (critical) and `200 lb scheduled Sep 16 — ships in 3 days` (info).
3. Late run is `warning`, never `critical`, even inside the critical window — the
   critical window belongs to "nobody is making it". A late run on an order that
   also has an uncovered shortage inside the critical window is still critical from
   rule 1; the late-run reason is listed as well.
4. Run overdue is `warning` on the **run** (a derived `flag: overdue` on
   `GET /production/runs`, §2j) and `info` on the **order**. It applies to status
   `planned` only; `in_progress` past its date is normal.
5. The `not short_lines` guards on overdue-with-stock and Not Ready to Ship
   (`:12507-12529`) keep their meaning with `short_lines` still defined on
   `shortage_lb` (covered or not): an order short on paper but fully scheduled is
   still not "stock on hand".
6. **Suppressed entirely:** the `"… lb not allocated …"` info sentence and the
   `" (allocations not enforced)"` note (`:12547-12574`). `_allocations_enforced()`
   is no longer read by `compute_so_health()`. `info_detail` keeps its key and
   becomes the rows behind the shortage sentences:
   `[{line_id, sku, product_name, short_lb, covered_lb, uncovered_lb, run_ids}]`.
   Dashboard: delete the regex at `so-list.js:28`, the detection at
   `dashboard.js:3230` and the banner at `:3251`; re-point the popovers at
   `:3266-3267` and `so-list.js:30` to the new `info_detail`. Merge criterion:
   `grep -rn "allocations not enforced" main.py dashboard/ tests/` returns nothing.
7. Unchanged: the clock, both env windows, `SO_HEALTH_READY_DAYS`, the
   `_fmt_number` formatter (STATUS-006), `product name (SKU)` labels, closed and
   cancelled orders silent.

Not changed by this spec but flagged (Q3): `coverable` still subtracts
`allocated_others` (`:12022`), so a *different* order's reservation can still push
this order into `critical`; and `inventory_ready` (`:12084-12087`) still requires
`allocated ≥ remaining`. Both are allocation-driven signals that survive
"allocations stop driving alarms" unless the owner says otherwise.

### 2i. Materials requirement, minimum version

Per run, derived at read time, returned as `materials: {status, items}` on
`GET /production/runs` when `?include=materials` (off by default; it is one BOM
walk per run):

```
1. routing: batch = product_bom component with p.type='batch'           (main.py:17351-17357 precedent)
            else products.parent_batch_product_id
            else → status 'unknown', items [], reason 'no batch routing'
   if run.product_id itself is type='batch' → batch = run.product_id
2. batches  = ceil(planned_qty_lb / batch.default_batch_lb)             (:17371 precedent)
            default_batch_lb NULL → 'unknown', reason 'no batch size'
3. for each batch_formulas row where NOT exclude_from_inventory:
     required_lb = quantity_lb × batches                                 (:17404 / :18195-18200 precedent)
     on_hand_lb  = posted SUM through POSTED_LINES for that ingredient   (:17413-17418 precedent)
     inbound_lb  = SUM(expected_qty − received) over open expected_receipts
                   with expected_date <= planned_date                    (remaining via EXPECTED_RECEIPT_RECEIVED_SQL :5613)
     short_lb    = max(0, required_lb − on_hand_lb − inbound_lb)
4. status = 'short' if any short_lb > 0 else 'ok'
   items  = [{ingredient_id, name, required_lb, on_hand_lb, inbound_lb, short_lb}]
```

Display string: `materials: ok` · `materials: short Oats 120 lb, Honey 30 lb` ·
`materials: unknown (no batch routing)`.

**Is the BOM complete enough to do this now?** For the batch → ingredient hop,
yes (25/25 active batch products, §1c). For the finished → batch hop, no: 34
active finished products have neither a `product_bom` batch component nor, in
some cases, a `parent_batch_product_id`, and the numbers on `product_bom.quantity`
are ignored by every existing consumer. So v1 **must** have the `unknown` state and
must not 404 or 500 on those SKUs the way `/production/requirements` does
(`:17359-17375`). Packaging is out of v1: no per-case packaging quantity exists to
multiply. Each run is compared against the same on-hand independently — two runs
of the same batch on the same day are not netted against each other in v1; the
daily board (§2j) is where a per-day netted view belongs later.

### 2j. Endpoints — dashboard-key allowlist only

All under `Depends(verify_api_key)`; added to `DASHBOARD_KEY_ALLOWLIST` with a
block comment like `:2669-2672`; **none added to any yaml**; every write stamps
`caller_source_tag(request)` into `created_by`/`updated_by`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/production/runs?from&to&status&product_id&include=materials,candidates` | List runs with per-run derived fields: `covered_lb`, `coverage: [{line_id, order_number, qty_lb}]`, `flag: overdue|late|null`, `actual_qty_lb`, optional `materials`, optional `candidate_actuals`. |
| `POST` | `/production/runs` | Create. Body `{product_id, planned_qty_lb, planned_date, line_id?, notes?, coverage?: [{sales_order_line_id, qty_lb}]}`. Rejects `products.no_production = true` and inactive products with 400. |
| `PATCH` | `/production/runs/{run_id}` | Update `planned_qty_lb`, `planned_date`, `line_id`, `notes`, `status` (`planned → in_progress → done`; `done` accepts `actual_transaction_id`). Shrinking `planned_qty_lb` below summed coverage → 409 `RUN_OVERCOVERED`. |
| `POST` | `/production/runs/{run_id}/cancel` | `status = 'cancelled'`, reason in `notes`. Coverage rows left in place. Cancelled is terminal. |
| `PUT` | `/production/runs/{run_id}/coverage` | Replace the run's coverage list atomically. Validates §2f invariants; 409 `RUN_OVERCOVERED`, 409 `LINE_PRODUCT_MISMATCH`, 409 `ORDER_NOT_OPEN`. |
| `DELETE` | `/production/runs/{run_id}/coverage/{sales_order_line_id}` | Clear one line. |
| `GET` | `/production/board?date=YYYY-MM-DD` | Daily board: `making` (runs with `planned_date = date`, not cancelled, plus that day's posted make/pack from the today-tile query `:17510`), `shipping` (open orders with `requested_ship_date = date`, fulfillment ≠ shipped, each with its `health`), `receiving` (open `expected_receipts` with `expected_date = date`, plus overdue ones flagged). Defaults to `_factory_today()`. |

Seven allowlist entries. Error shape follows the existing dict contract
(`FOLLOWUPS.md` §2). The 7-day `POST /schedule` stays master-key only and is not
touched.

### 2k. Lock-and-order inventory

Normative order (`main.py:464-476`, findings doc `:541-554`):
`sales_orders row → sales_order_lines rows → product/lot rows`, ascending id within
a step. `production_runs` is inserted as **step 2b**: after the order's lines, and
it never reaches step 3 — no path in this spec locks a product, lot or allocation
row.

| # | Path | Locks, in order | Strength | Notes |
|---|---|---|---|---|
| R1 | `POST /production/runs` without coverage | none | — | plain INSERT; FK `KEY SHARE` on `products`, `production_lines` |
| R2 | `POST /production/runs` with coverage, `PUT …/coverage` | (1) every distinct `sales_orders` row of the listed lines, **ascending order id**, `_lock_sales_order()` (`:477`) → (2) the listed `sales_order_lines` rows, ascending, `_lock_sales_order_lines()` (`:500`) → (2b) the `production_runs` row → implicit locks on `run_coverage` rows | `FOR NO KEY UPDATE` throughout | Locking several orders ascending cannot cycle with any single-order 1→2→3 writer, nor with another R2. State check `state='open'` runs under the order lock, as `_load_allocatable_line` does (`:1579-1592`). |
| R3 | `DELETE …/coverage/{line_id}` | (1) that line's order → (2) that line → (2b) the run | `FOR NO KEY UPDATE` | same as R2 with one line |
| R4 | `PATCH /production/runs/{id}` (qty/date/line/notes/status) | (2b) the run row only | `FOR NO KEY UPDATE` | reads `run_coverage` SUM under the run lock; **never** reaches back for an order or line (the A16 pattern, findings `:605-612`) |
| R5 | `POST …/cancel` | (2b) the run row only | `FOR NO KEY UPDATE` | as R4 |
| R6 | `status → done` with `actual_transaction_id` | (2b) the run row; transaction read unlocked through `POSTED_LINES` | — | no `lots` lock; the ledger row is immutable |
| R7 | Health / list / detail / board reads | none | — | the coverage CTE joins `run_coverage` and `production_runs` unlocked, like D9 (`:12108`) |
| R8 | SO exits (`close`/`cancel`/`reopen`), `ship_order`, allocation writes | **unchanged** | — | they do not read or write `run_coverage`; stale coverage is filtered on read |
| R9 | `/make`, `/pack` | **unchanged** | — | never touch `production_runs` (§2g) |

Deadlock argument: every writer that holds a run lock either holds nothing else
(R4–R6) or acquired it strictly after its order and line locks (R2, R3). No
writer holds a run lock and then waits on an order, line, product or lot. The
existing 1→2→3 writers never wait on a run. Hence no cycle through step 2b.

Tests to add, mirroring `tests/test_sales_order_state_model.py:3174-4266`: a
race where R2 on order O and `cancel_order` on O interleave and neither
deadlocks; a source-order fingerprint that `PUT …/coverage` calls
`_lock_sales_order` before `_lock_sales_order_lines` before the run `SELECT … FOR NO
KEY UPDATE`; and the allowlist-shape tests in `test_dashboard_api_key.py`.

### 2l. Migration 053 sketch

`migrations/053_production_runs.sql`, applied by hand **before** the code that
reads it merges (051/052 precedent, `FACTORY_LEDGER_CHANGELOG.md` rows 131/142):

```
psql "$(cat ~/.config/factory-ledger/db_url)" -v ON_ERROR_STOP=1 -f migrations/053_production_runs.sql
```

`db_url` must be the **session pooler (port 5432)** URL, never 6543
(`CLAUDE.md` hard rule; `scripts/psql_ro.sh:32`). The file is wrapped in
`BEGIN; … COMMIT;` like 051 (`051:31`), not editor-style like 052, because psql
is the applier. After apply: `scripts/dump_prod_schema.sh` to refresh
`tests/schema/schema.sql`, and a changelog row.

```sql
-- 053: production_runs + run_coverage (docs/design/scheduling-spec-draft.md)
-- Idempotent and re-runnable. No seed rows. No backfill.
BEGIN;

CREATE TABLE IF NOT EXISTS public.production_runs (
    id                    serial PRIMARY KEY,
    product_id            integer NOT NULL REFERENCES public.products(id),
    planned_qty_lb        numeric(14,4) NOT NULL,
    planned_date          date NOT NULL,
    line_id               integer REFERENCES public.production_lines(id),
    status                text NOT NULL DEFAULT 'planned',
    notes                 text,
    actual_transaction_id integer REFERENCES public.transactions(id),
    created_at            timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_at_source     text NOT NULL DEFAULT 'database',
    created_by            text,
    updated_at            timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_by            text
);

CREATE TABLE IF NOT EXISTS public.run_coverage (
    run_id              integer NOT NULL REFERENCES public.production_runs(id),
    sales_order_line_id integer NOT NULL REFERENCES public.sales_order_lines(id),
    qty_lb              numeric(14,4) NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_by          text,
    PRIMARY KEY (run_id, sales_order_line_id)
);

-- Constraints added through catalog checks so a partial earlier run does not fail (052 pattern, :58-87).
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='production_runs_qty_check') THEN
    ALTER TABLE public.production_runs ADD CONSTRAINT production_runs_qty_check CHECK (planned_qty_lb > 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='production_runs_status_check') THEN
    ALTER TABLE public.production_runs ADD CONSTRAINT production_runs_status_check
      CHECK (status IN ('planned','in_progress','done','cancelled'));
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='run_coverage_qty_check') THEN
    ALTER TABLE public.run_coverage ADD CONSTRAINT run_coverage_qty_check CHECK (qty_lb > 0);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_production_runs_planned_date ON public.production_runs (planned_date);
CREATE INDEX IF NOT EXISTS idx_production_runs_product_status ON public.production_runs (product_id, status);
CREATE INDEX IF NOT EXISTS idx_run_coverage_line ON public.run_coverage (sales_order_line_id);

-- created_at / created_at_source immutability, same trigger production_schedule uses (039).
DROP TRIGGER IF EXISTS trg_production_runs_created_at ON public.production_runs;
CREATE TRIGGER trg_production_runs_created_at BEFORE INSERT OR UPDATE ON public.production_runs
    FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();
DROP TRIGGER IF EXISTS trg_run_coverage_created_at ON public.run_coverage;
CREATE TRIGGER trg_run_coverage_created_at BEFORE INSERT OR UPDATE ON public.run_coverage
    FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();

-- Marker (052 pattern, :89-124). migration_markers exists since 051; created here too for a fresh DB.
CREATE TABLE IF NOT EXISTS public.migration_markers (
    name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT clock_timestamp());
INSERT INTO public.migration_markers (name) VALUES ('053_production_runs') ON CONFLICT (name) DO NOTHING;

COMMIT;
```

`updated_at` is maintained by the application on every UPDATE (no trigger; the
`ledger_enforce_created_at` trigger only guards `created_at`). A `down` file goes
in `migrations/down/` per the existing convention. Verify with
`select * from migration_markers where name = '053_production_runs';`.

---

## Part 3 — plan

Size target is PR #46 (+178/−12, 2 files), not PR #40 (+5,330, 6 files). Realistic
sizes below include tests, which is where the state-model PRs put most of their
lines; each PR is one reviewable concern. Estimates are working days of one
person including Codex cross-review; the migration apply itself is minutes.

| # | PR | Contents | Merge criterion | Est. |
|---|---|---|---|---|
| S1 | `feat/053-production-runs` — schema + endpoints | Migration 053; `production_runs`/`run_coverage` models; the six run/coverage routes from §2j; allowlist entries; `caller_source_tag` attribution; lock sequence R1–R6; tests: CRUD, invariants (409s), attribution (dashboard key → `'dashboard'`, actor key → name), lock fingerprint, one deadlock race, allowlist shape. **No Health change.** | 053 applied to prod via psql and schema dump refreshed before merge; all new routes 403 for master-key-less callers and 200 for dashboard + actor keys; `grep -c operationId openapi-gpt-v3.yaml` still 30; full suite green. | 2 days (~500 lines, half tests) |
| S2 | `feat/so-health-v3` — coverage replaces allocations | `coverage` CTE in `SALES_ORDER_READINESS_SQL`; new `_line_readiness` fields; `compute_so_health()` rules 1–7 of §2h; `info_detail` re-shaped; dashboard string/popover changes at the six sites in §1b; tier-matrix tests for every row and both sides of each boundary; mutation check on the `, not scheduled` split; findings-doc section "Health v3" in the v2.1 format. | `grep -rn "allocations not enforced"` empty; tier matrix green; existing v2.1 clock/window tests untouched and green; dashboard list + detail render the new info sentences. | 2 days (~700 lines) |
| S3 | `feat/run-materials` — materials derived field | `?include=materials` on `GET /production/runs`; routing/`unknown` rules of §2i; inbound from open expected receipts; tests for the three routing cases (product_bom, parent_batch_product_id, none), `exclude_from_inventory`, expected receipts before/after `planned_date`. | Every active finished product returns `ok`/`short`/`unknown`, never 4xx/5xx; numbers match `/production/requirements` where that endpoint works. | 1 day |
| S4a | `feat/production-board-api` | `GET /production/board` per §2j, reusing the today-tile day window and `_factory_today()`. | Three lists agree with `/production/runs`, `/sales/orders?…`, `/expected-receipts?…` for a fixed date in tests. | 0.5 day |
| S4b | `feat/production-board-ui` — dashboard | Runs list + create/edit/cancel modal; coverage picker from an open order's short lines; daily board tab; Health chips unchanged (server-owned). Static assets only. | Visual audit passes the STATUS rules for the new screens; asset versions bumped. | 2 days |
| S5 (later) | `chore/retire-schedule-engine` | Remove `POST /schedule`, the 7-day engine, `production_schedule` (drop after export), `scheduling_config`; keep `production_lines` and `product_line_assignments`. Separate owner decision. | — | 0.5 day |

Order: S1 → S2 → S3 → S4a → S4b. S2 is the one that changes what the floor sees;
S1 alone changes nothing visible, which is the point of merging it first. S3 and
S4a are independent of each other once S1 is in.

---

## Decisions needed from the owner before implementation

1. **New tables or reuse `production_schedule`?** This spec proposes new
   `production_runs` + `run_coverage` and leaves the migration-004 tables and
   `POST /schedule` dormant, to be retired in S5. Confirm, or direct reuse (which
   means dropping its unique key and NOT NULLs and breaking the dormant `confirm`
   action).
2. **What is a run?** Proposed: one **finished SKU** on one day (the pack-out that
   ships), so coverage can point at SO lines directly; batch count lives in
   `notes`. Alternative: runs are bake batches and coverage walks `product_bom` —
   more faithful to the oven, but the FG→batch routing is missing for 34 active
   SKUs today.
3. **How far do allocations retreat?** Proposed minimum: delete the "not
   allocated" info sentence and the "(allocations not enforced)" note. Two
   allocation effects survive that: another order's reservation still reduces
   this order's `coverable` (`main.py:12022`), and `inventory_ready` still requires
   `allocated ≥ remaining` (`:12084-12087`). Keep both, drop both, or drop only the
   second?
4. **Manual completion only?** Proposed: the ledger never marks a run done;
   the board shows a candidate posted make/pack and the operator confirms. Confirm
   that no automatic product + date matching should flip a run to `done`.
5. **Who may plan?** The allowlist admits the shared dashboard key and every
   actor key regardless of `role`. Is any floor-role holder allowed to create,
   cover and cancel runs, or should the six write routes wait for per-role scoping
   (deferred in #49, `so-state-model-findings.md:1225-1240`)? Related: is
   "shipping today" on the board `requested_ship_date = today` for open, unshipped
   orders, and should overdue orders appear there too?

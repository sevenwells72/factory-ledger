# S1 amendment — bake runs vs pack runs

Date: 2026-09-15. Status: **DRAFT for owner review. No code, no migration, nothing applied.**
Updated 2026-09-15 (same day): coconut pan weight ruled **360 lb** by the owner; §2b, §2d and
open questions 2 and 11 reflect it.
Base: `main` @ `22b606d` (S1 merged in #52, S2 merged in #54, Runs screen in #53/#55;
migration 053 applied to production 2026-09-14 17:07 UTC).

Amends: `docs/design/scheduling-spec-draft.md` Part 4 (S1 as built) and
`migrations/053_production_runs.sql`. Where this document and Part 4 differ, this
document wins once the owner accepts it; until then Part 4 is normative.

Out of scope, by instruction: `feat/planner-v2` (read for the yield survey only, not
touched), `production_schedule` / `POST /schedule` (migration 004, still dormant and
untouched), `openapi-gpt-v3.yaml` (stays at 30 operations; none of these routes are
in any yaml).

Every line number below is from `main` @ `22b606d`; grep by symbol before editing.

---

## 0. What changes, in one table

| S1 as built (Part 4) | S1 amended (this doc) |
|---|---|
| A run is one **finished SKU** on one day. `POST` rejects `type <> 'finished'` (`_load_schedulable_product`, `main.py:18272`). | A run has a **`run_type`**: `bake`, `pack`, `coconut`, `other`. Bake and coconut runs yield a **batch product** (the recipe / WIP item); pack runs yield a finished SKU exactly as today. |
| Quantity as entered: `planned_unit IN ('cases','lb')`, `case_size_lb_used` records the multiplier. | Adds `'pans'`; `pan_yield_lb_used` records the per-pan multiplier the same way. `planned_qty_lb` stays canonical and, for bake/coconut runs, **is the expected WIP pounds**. |
| Coverage requires `line.product_id = run.product_id` (`409 LINE_PRODUCT_MISMATCH`). | For bake/coconut runs the line's finished SKU must **route to** the run's batch product (`parent_batch_product_id` or `product_bom` batch component). Same table, same columns, same "soft" meaning. |
| Form has "Production line number (optional)". | Field removed. Line is inferred server-side from the item, with a run-type fallback. Column stays nullable, label only. |
| Migration 053. | Migration 054, additive only (two columns, one widened CHECK). `run_coverage` unchanged. |

Nothing here changes what Health v3 reads: the `coverage` CTE (`main.py:12058-12075`)
joins `run_coverage` by `sales_order_line_id` and never looks at the run's product,
so bake-run coverage flows into `covered_lb` with **zero S2 code change**. See §4.

---

## 1. `run_type` on `production_runs`

### 1a. Values

| `run_type` | Item (`product_id`) | Native unit(s) | `planned_qty_lb` means | Line (inferred, §6) | Evidence (`GET …/evidence`) | Coverage rule (§4) |
|---|---|---|---|---|---|---|
| `bake` | `products.type = 'batch'`, granola family | `pans` (or `lb`) | **expected WIP lb** = pans × per-pan yield | `granola` | posted `make` of the batch product | line's finished SKU routes to this batch |
| `coconut` | `products.type = 'batch'`, coconut family (90003, 90004, 90005, 90007) | `pans` (or `lb`) | expected WIP lb = pans × **360** (sweetened) / 300 (toasted); the stored `yield_multiplier = 1.11` is **not** applied (§2d) | `coconut` | posted `make` of the batch product | same as bake |
| `pack` | `products.type = 'finished'`, not `no_production`, not service — **exactly S1 today** | `cases` or `lb` | finished lb (cases × `case_size_lb`) | `pouch` if `pack_format = 'bagged'`, else `bulk_pack` | posted `pack` of the finished SKU | `line.product_id = run.product_id` (unchanged) |
| `other` | any active, non-service product the factory handles (graham repack 31012, chips 25013/25014, Kookies & Kreme 10301, coconut chips 70051 — the five active producible finished SKUs with no batch routing) | `lb` (or `cases` when `case_size_lb` exists) | lb | none (NULL) | posted `make` **or** `pack` of the product (today's rule) | equality, as pack |

Why `coconut` is its own value and not `bake`: same mechanics (pans of a batch
product, `make` evidence) but a different line, a stored hydration multiplier that the
owner has ruled must not be applied (`yield_multiplier = 1.11` on 90003/90004/90005, §2d),
two cycles a day, and toasted pans
that stay locked until emptied the next workday
(`dashboard/scheduler/seven-wells-production-board.html:290-302`). Health never looks
at `run_type`; the board and the line inference do. Collapsing it into `bake` would
put "which oven" into `notes`, which is the WhatsApp problem again.

Why `other` exists: five active SKUs are made or repacked here with no batch step
(list above, verified live 2026-09-15). Without `other` they cannot be scheduled at
all after the finished-only check is replaced by a run-type-aware one.

### 1b. Representation

`run_type text NOT NULL` with a `CHECK (run_type IN ('bake','pack','coconut','other'))`
— **not** a Postgres `CREATE TYPE … AS ENUM`. Every enum-shaped column in this schema
is text + CHECK (`production_runs.status`, `products.type` `schema.sql:1397`,
`trace_events.event_type` `048:39-40`, `expected_receipts.status`); a native enum
needs `ALTER TYPE … ADD VALUE`, which cannot run inside the same transaction as a
statement that uses the new value and cannot be removed, so it is strictly worse for
a rollback (§7).

DB default `'pack'`. The API **always** sends `run_type` explicitly (a missing field is
`422`); the default exists only so that a Railway rollback to the pre-054 build, whose
`INSERT` does not name the column, keeps working (§7d). Every S1 row that exists at
apply time is a finished-SKU run, i.e. a pack run, so the backfill is the default.

Constants in `main.py`: `PRODUCTION_RUN_TYPES = ("bake", "pack", "coconut", "other")`,
plus a per-type table of allowed item types and allowed units, mirrored in the runs
screen (no server round-trip to decide which unit options to show).

### 1c. Status vocabulary unchanged

`planned → in_progress → done | cancelled` stays for every type. `run_type` is
orthogonal to `status`; no new terminal state. `PATCH` may not change `run_type`
(cancel and recreate, as with `product_id`).

---

## 2. Bake runs: recipe item, pans, expected pounds

### 2a. Item and quantity

`product_id` references a `products.type = 'batch'` row. There are 25, all active, all
with `default_batch_lb` (live query 2026-09-15; `SYSTEM_KNOWLEDGE.md:325` agrees).
`_load_schedulable_product` (`main.py:18272`) becomes run-type-aware: for
`bake`/`coconut` it requires `type = 'batch'`; for `pack` it keeps today's four checks;
for `other` it requires active and non-service only. Error code stays
`400 PRODUCT_NOT_SCHEDULABLE` with the reason text naming the run type.

Quantity is entered in **pans**: `planned_qty = 12`, `planned_unit = 'pans'`. The API
converts to canonical pounds exactly as `_run_quantity_lb` (`main.py:18232`) converts
cases today:

```
per_pan_lb      = products.default_batch_lb                            -- yield_multiplier NOT applied, see §2d
planned_qty_lb  = _run_lb(Decimal(pans) × Decimal(per_pan_lb))        -- numeric(14,4), ROUND_HALF_UP
pan_yield_lb_used = per_pan_lb                                        -- stored, audit trail
```

`yield_multiplier` is deliberately left out. For every granola batch it is `1.0`, so the
result is identical to what `/make` posts; for the three sweetened-coconut batches it is
`1.11`, and the owner has ruled (2026-09-15) that **360 lb is the real pan weight**, so a
run must plan 360 per pan even though `/make` currently posts 399.6 (§2d). When `/make`
is corrected, the two agree again without touching this rule.

`planned_qty_lb` keeps its column name (canonical pounds, the thing coverage sums
against). The API response adds `expected_lb` as an alias of `planned_qty_lb` on
bake/coconut runs so the screen can say "12 pans · expected 3,876 lb" without the
word "planned" attached to a number nobody typed. A batch product with no
`default_batch_lb` (none today) gets `400 PAN_YIELD_REQUIRED`, the pan twin of
`CASE_WEIGHT_REQUIRED`; the run can still be planned in `lb`.

`PATCH` re-converts when `planned_qty`/`planned_unit` change, using the **current**
product yield, and leaves `pan_yield_lb_used` alone when they do not — the same
"omit unchanged native fields" rule the screen already applies to cases
(`dashboard/runs.js:119-121`).

### 2b. Does a per-pan yield already exist? Yes — do not add one

Survey of every place a pan or per-pan number lives:

| Where | What it is | Verdict |
|---|---|---|
| **`products.default_batch_lb × products.yield_multiplier`** (`schema.sql`, products) | What `POST /make` posts as output per batch (`total_output = default_batch_lb × batches × yield_multiplier`, spec §1a). `_made_unit_size_lbs()` (`main.py:15990`) is documented as "Finished output weight of one physical batch/pan"; the production calendar derives `batch_count` from it (`main.py:16069-16076`); the batch-inventory tile calls it "batch/pan counts" (`:16199`). The 2026-08-24 data-health baseline reconciles "coconut pans (batches)" one-for-one against the floor's form (`docs/data-health-baseline-2026-08-24.md:882`). **In this codebase one batch is one pan.** | **This is the per-pan yield. Use it.** |
| `_ORDERS_MATRIX_PAN_YIELD` (`main.py:11451`) | Hard-coded dict keyed by **finished** SKU, "Source: CNS Production Source of Truth v1.0, 2026-07-02", used only by the orders-matrix export to print "≈ N pans @ X lb/pan". | Duplicate, drifted: 322.6 vs stored 323 (90002); 380.12 vs 380; 370.12 vs 370; coconut **360** = pre-hydration `default_batch_lb`, whereas `/make` posts 360 × 1.11 = **399.6** per pan. Not a source of truth; retire it later (separate chore). |
| `dashboard/scheduler/seven-wells-production-board.html` `CATALOG` (`:281-349`) | `pl` "base lbs per pan (bake)", `fl` "finished lbs per pan (post-premix)", `cpp` cases per pan (coconut), plus a `pan` name shared across SKUs baked from one batch (branch `feat/pan-product-mapping`, `bd7bade`, touches only this file). Never reads or writes the API. | Same numbers as the matrix dict, with placeholders flagged in-line ("BATCH UNMAPPED — 330 lb/pan placeholder"). Useful as a cross-check, not as data. |
| `feat/planner-v2` branch | `demand_planning_v1.sql`, `render_planner_v2.py`, `render_demand_plan.py`; migrations stop at 035. | **No pan or yield table.** Nothing to reuse; branch untouched. |
| `boms.expected_yield_pct`, `bom_lines.quantity_per_batch` (`schema.sql:590-631`) | Dead tables, zero rows, no reader (spec §1c). | Ignore. |
| `product_bom.quantity` | Ignored by every consumer (`SYSTEM_KNOWLEDGE.md:1391`). | Ignore. |
| `products.yield_25lb_cases / yield_10lb_cases / yield_retail_cases / yield_retail_bags` | Legacy per-batch case yields, populated for some products, read by nothing on the run path. | Out of scope; could later derive "cases per pan" for the board. |

**Conclusion:** nothing must be added to `products`, `batch_formulas` or any planner
table. The one addition is per-run: `production_runs.pan_yield_lb_used`, the value the
API multiplied by, for the same reason `case_size_lb_used` exists — a later change to
`default_batch_lb` must not silently re-price an existing plan.

Live per-pan values the API would use (batch → `default_batch_lb`):
90002 Classic #9 **323**; 90001 Classic Choc Chip #9 **348**; 90010 Vanilla Almond
**380**; 90024 Vanilla Crisp #16 **370**; 90016 SS Original #1 **350**; 90011 SS Choc
Chip #2 **393**; 90003/90004/90005 sweetened coconut **360** (owner ruling; the stored
1.11 multiplier is not applied, §2d); 90007 toasted coconut **300**; 95005 BS PB Banana
**452**; 90008 Fruit Nut **384.52**.

One data fact still to settle, not blocking (open question 3): the matrix dict's 322.6
vs the stored 323 is a 0.1 % disagreement about the same pan. The coconut basis was
settled the same day this draft was written: **360 lb per pan** (open question 2,
resolved). What that means for the ledger, and the caveat it leaves on coconut
evidence, is §2d.

Known anomaly carried forward unchanged: 90008 Granola Fruit Nut Batch has formula
rows totalling 25 lb against `default_batch_lb = 384.52` (`SYSTEM_KNOWLEDGE.md:517`).
It does not affect the pan yield (which is `default_batch_lb`), only S3 materials.

### 2c. Evidence for bake runs

`_run_evidence` (`main.py:18601`) already queries posted `make`/`pack` output lines
for `run.product_id` in `[planned_date − 1, planned_date + 1]`. For a batch product
only `make` ever posts a positive line, so it works unchanged; pin it anyway: filter
`t.type = 'make'` for bake/coconut, `'pack'` for pack, both for other.
`recorded_qty` in the run's native unit becomes `recorded_lb / pan_yield_lb_used`
when the unit is pans (today it divides by `case_size_lb_used`). `suggested_state`
thresholds unchanged. Completion stays human-confirmed, writes the run row only
(Part 4 §4a decision 4 stands).

**Coconut caveat until `/make` is corrected (§2d):** a sweetened-coconut run planned at
12 pans expects 4,320 lb, but a 12-batch `/make` posts 4,795.2 lb today. Evidence will
therefore read `looks_complete` at 11 pans posted and `recorded_qty` will show 13.3 pans
for a 12-pan make. This is a display artefact of the ledger's multiplier, not of the
run; it disappears once the three products' `yield_multiplier` is set to 1.0 (open
question 11). Do not "fix" it by dividing evidence by 1.11 in the run code — that
would hard-code the error the owner has just ruled against.

### 2d. Coconut pan weight — what the ledger shows (read-only, 2026-09-15)

Owner ruling: **360 lb is the real weight of one sweetened-coconut pan.** Findings that
sit behind the rule and behind open question 11:

* **As stored:** 90003, 90004, 90005 have `default_batch_lb = 360`, `yield_multiplier =
  1.11` (product of 399.6); 90007 toasted has 300 and 1.0. The 1.11 is entered data,
  not a column default (`FACTORY_LEDGER_CHANGELOG.md` row 70 confirms it "is entered
  data on the three sweetened-coconut batches"). The formulas total 389.5 lb including
  65 lb of `exclude_from_inventory` water, 324.5 lb without it.
* **What `/make` posts:** `default_batch_lb × batches × yield_multiplier`
  (`main.py:8252-8254`); `formula_weight_lb` there is `default_batch_lb × batches`, not
  the formula rows. So every sweetened-coconut make since the multiplier was entered
  posts **399.6 lb per pan**.
* **Lots and pounds, all time, posted:** 90003 20 lots / 21 makes / 32,368 lb; 90004
  107 lots / 107 makes / 397,238 lb; 90005 13 lots / 14 makes / 15,185 lb; 90007 26
  lots / 25 makes / 41,400 lb (one void). 131 of the 142 sweetened makes are exact
  multiples of 399.6; the other 11 are 90004's first makes, 2026-02-05 to 02-16, exact
  multiples of 360 (32,400 lb) — the multiplier was entered around 2026-02-16/17. All 25
  toasted makes are exact multiples of 300.
* **Pounds booked above the 360 basis:** 90003 3,208 lb; 90004 36,155 lb; 90005 1,505 lb;
  **40,868 lb in total** (posted at 399.6 ÷ 1.11).
* **Pan counts are right; pounds are not.** Every dashboard reader that shows "pans"
  divides posted pounds by `default_batch_lb × yield_multiplier` (`_made_unit_size_lbs`
  `main.py:15990`; production calendar `:16069-16076`; batch-inventory tile
  `:16265-16266`; today-tile `:17847`), so 4,795.2 lb reads as 12 pans — which is why the
  2026-08-24 baseline found every coconut day from Aug 3 matching the floor's form
  pan-for-pan (`docs/data-health-baseline-2026-08-24.md:1201`). Nothing anywhere
  reconciles the **pounds** against 360: the Aug 14 physical count explicitly excluded
  the coconut batch silos (`docs/audits/physical-count-2026-08-14.md:6`), and the
  variance recon held them as "no floor, no consume"
  (`docs/audits/inventory-variance-recon-plan.md:278`). The board catalog is the one
  artefact on the 360 basis: `cpp: 36` cases of 10 lb per pan = 360 lb
  (`seven-wells-production-board.html:291-299`).
* **Where 399.6 / 1.11 appear:** code — `/make` output (`main.py:8091-8093`, `:8252-8254`),
  `_made_unit_size_lbs` and its four callers above, `PUT /admin/products` accepts
  `yield_multiplier` (`:17084-17086`), the startup migration-005 block (`:2119-2127`);
  tests — `tests/test_dashboard_b2.py:96-100` and
  `tests/test_dashboard_production_calendar.py:110-113` pin `1.11 → 399.6 → 12 pans`
  on fixture data; docs — `FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:481, :620, :1003-1016,
  :1094, :1155`, `FACTORY_LEDGER_CHANGELOG.md` rows 54 and 70, `CONTEXT.md:87, :185,
  :263`; board catalog — none (it is on 360). `_ORDERS_MATRIX_PAN_YIELD` says 360.
* **A caution for the correction (open question 11).** Coconut batch stock is
  consumed by `/pack` at real case weights (10 lb × 36 per pan = 360), so the pack side
  is already on the 360 basis. Re-basing the historical makes to 360 without touching
  anything else would drive two products' cumulative balance negative: 90004 to about
  **−10,300 lb** (328,683 re-based + 32,400 early makes − 30,780 net adjustments
  − 340,595 packed) and 90005 to about **−740 lb**; 90003 stays positive (+560). The
  30,780 lb of net adjustments on 90004 are all February go-live cleanup
  ("Adjustment: −3600.0 lb" style, `legacy-unattributed`) plus one 60 lb audit fix,
  none of them yield corrections. Current posted on-hand is 90003 3,768 lb, 90004
  25,863 lb, 90005 762 lb, 90007 500 lb. So either the floor has been under-counting
  pans into `/make`, or more was packed than the 360 basis allows, or the silos hold
  less than the ledger says — a physical count of the coconut silos is the only thing
  that settles which, and it should come before any historical re-basing.

---

## 3. WIP representation: reuse the batch product, add nothing

**Proposal: the WIP item is the `products.type = 'batch'` row itself.** "Granola #9 –
bulk" is product 107, `90002 Batch Classic Granola #9`. Its on-hand WIP is the posted
balance of that product through `POSTED_LINES`; its lots are the 374
`entry_source = 'production_output'` lots `/make` has already created (live count
2026-09-15); `/pack` already consumes them FIFO. A bake run's `product_id` points at
it. **Zero new schema, zero new product rows.**

Rejected alternatives:

| Alternative | Why not |
|---|---|
| A new product row per recipe, e.g. "Granola #9 – bulk" as a second `finished` or `ingredient` item | Forks the WIP identity: `/make` would keep posting to 90002 and `/pack` would keep consuming 90002, so the new row would never carry a pound and every evidence/on-hand read would need a mapping table. |
| A `wip_items` table or a `products.is_wip` flag | `type = 'batch'` **is** the flag; every reader that matters (`/make`, `/pack`, `fifo_lot_balances`, the batch-inventory tile) already keys on it. |
| Storing "uncovered WIP" as a balance | Same mistake as a stored remaining on expected receipts (`041:14-18`): it is `planned_qty_lb − Σ run_coverage.qty_lb`, derived at read time, and it is **not inventory** (§4c). |

Display: the screen labels a bake run by the batch product's name plus pans and
expected pounds ("Batch Classic Granola #9 · 12 pans · ≈ 3,876 lb"). Any "bulk"
wording is presentation only.

One precedent to leave alone: product 291 `15999 WIP Banana / PB-chip mix (for PBB)`
is `type = 'ingredient'` with `parent_batch_product_id = 122` — a pre-mix modelled as
an ingredient. It is not a bake output and needs no run type; if the floor ever wants
to schedule it, `other` covers it.

---

## 4. Coverage on bake runs — soft, same table

### 4a. Table unchanged

`run_coverage (run_id, sales_order_line_id, qty_lb, created_at, created_by)` stays
exactly as 053 built it. `qty_lb` is **finished pounds of the sales-order line**, as
today; 1 lb of WIP is taken as 1 lb of finished product because `/pack` posts the
FG line at the batch pounds consumed (pack add-ins are ignored in S1; open question 5).

### 4b. What changes in `PUT /production/runs/{id}/coverage` (`main.py:18721`)

Only the product check. Today: `sol.product_id = run.product_id` else
`409 LINE_PRODUCT_MISMATCH`. Amended, by run type:

* `pack`, `other`: unchanged.
* `bake`, `coconut`: the line's finished product must **route to** the run's batch
  product — `products.parent_batch_product_id = run.product_id` **or** a `product_bom`
  row with `finished_product_id = sol.product_id` and `component_product_id =
  run.product_id` where the component is `type = 'batch'`. Same error code, message
  "line's product is not packed from this batch". Live coverage of that rule
  (2026-09-15): of 52 active producible finished SKUs, 47 have a parent, 37 have a
  batch BOM row, the two never disagree where both exist, and the 5 with neither are
  exactly the `other` list in §1a.

Everything else in the PUT is unchanged: order `state = 'open'`, line open and not a
service line, `qty_lb ≤` effective remaining (`409 COVERAGE_EXCEEDS_REMAINING`),
`Σ qty_lb ≤ planned_qty_lb` (`409 RUN_OVERCOVERED` — for a bake run that reads "you
cannot promise more finished pounds than the pans will yield"), run active. Lock
order is unchanged: orders ascending → lines ascending → run (step 2b); the products
row for the routing check is read unlocked, as `_load_schedulable_product` reads it
today. No path locks a product, lot or allocation row.

### 4c. Why it is "soft", and what that means for Health

Coverage is a statement of intent — "this bake is expected to feed this line" — with
no reservation, no allocation row, no effect on availability and no lock on stock.
That is already the S2 rule for finished-run coverage (`so-state-model-findings.md`
§ Health v3, "Coverage is not availability"). Bake coverage inherits it unchanged.

Health v3 needs **no change** to see bake coverage: the `coverage` CTE joins
`run_coverage` by line id and filters on `r.status IN ('planned','in_progress')`
(`main.py:12058-12075`). A short line covered by a bake run reads `Short 500 lb —
covered by run on Sep 16` today, verbatim. Whether the sentence should say *bake* is
open question 6.

**Uncovered WIP** per run = `planned_qty_lb − covered_lb`, returned by
`GET /production/runs` as `uncovered_lb` (derived in `_serialize_production_run`,
`main.py:18367`, next to the existing `covered_lb`). It is never stored and never
enters `on_hand`, `available_lb` or the §4c waterfall — it is pounds nobody has
promised yet, not pounds that exist. A per-day, per-batch-product roll-up ("Classic #9:
36 pans planned, 9,200 lb promised, 2,428 lb uncovered") is a board feature for S4a,
computed from the same rows.

Several bake runs may cover one line, and one bake run may cover lines of several
finished SKUs from the same batch (10 lb and 25 lb Classic #9 out of one bake) —
that is the point of covering at the batch level.

---

## 5. Pack runs: defined now, wired in S2

### 5a. Definition

`run_type = 'pack'`: one finished SKU, planned in cases (or lb), `planned_qty_lb =
cases × case_size_lb_used`, `pouch`/`bulk_pack` line, `pack` evidence, coverage by
product equality. **This is the S1 run as built, renamed.** Every S1 endpoint,
validation, lock row (A26–A30) and test keeps working for it; the only S1-amendment
change a pack run sees is that it now carries `run_type = 'pack'`.

What S2 adds, and what this amendment therefore leaves open: **a pack run consumes
WIP.** S2 decides whether that is declared (a `run_inputs (pack_run_id, batch_product_id
or bake_run_id, lb)` table) or derived (the posted pack transaction's negative batch
lines, read through `POSTED_LINES` the way evidence is), and how a pack run's coverage
relates to the bake run's coverage on the same line (supersede, or split).

### 5b. What S1-amended must NOT do, because it would block S2

1. **No `parent_run_id`, `consumes_run_id`, `source_batch_product_id` or `wip_lb_*`
   column on `production_runs`.** Part 4 §4a decision 2 already forbids it; S2 picks
   link-table vs ledger-derivation with data in hand, and a nullable column added now
   is the shape S2 would have to undo.
2. **No second WIP product rows** (§3). Consumption in S2 has to point at the product
   `/pack` actually debits.
3. **Uncovered WIP is never availability.** Do not net it against pack-run demand,
   against the batch product's `on_hand`, or into the §4c waterfall. Availability is
   the posted ledger only (Part 4 §4c rule 5).
4. **Keep `run_coverage` product-agnostic and keep the Health CTE joining by line id.**
   Do not add `run_type` to the CTE or delete a bake run's coverage when a pack run is
   created for the same line. The double-count rule ("a pack run fed by this bake
   supersedes the bake's coverage on that line") is S2's to write, once consumption
   exists to key it on. Until then the screen should warn when a line is covered by
   both a bake and a pack run; Health may over-count `covered_lb` on that line, which
   errs toward *quiet*, never toward a false alarm.
5. **`pan_yield_lb_used` stays nullable; `case_size_lb_used` and `'cases'` stay.** Pack
   runs have no pan; bake runs have no case. No `NOT NULL`, no `CHECK` that ties a
   unit to a run type at the DB level — the API enforces "bake ⇒ pans or lb, pack ⇒
   cases or lb", so S2 can relax either without a migration.
6. **Do not replace the finished-only product check with "anything goes".** Pack runs
   still need active / finished / not `no_production` / not service; the check becomes
   run-type-aware, not weaker.
7. **`/make` and `/pack` stay untouched (R9); completion creates nothing.** No auto-flip
   of a bake run to `done` on a posted make, no lot creation on complete, no
   `production_runs` write from a ledger path. S2's consumption link, whichever form,
   must also be able to rely on this.
8. **Do not seed `product_line_assignments` or change `production_lines` in 054.** Line
   is a label (§6); S2 pack-line logic must not depend on rows this migration invented.
9. **No route in any GPT yaml.** Still 30 operations.

---

## 6. Production line: off the form, inferred from the item

### 6a. Form

Remove the "Production line number (optional)" input and its helper sentence
(`dashboard/runs.js:113-114`), the prefill from product search (`:136`), and `line_id`
from both the POST and PATCH bodies (`:118`). Keep the row display of
`line_name || line_code` (`:69`) and keep the `PRODUCTION_LINE_NOT_FOUND` error string
harmlessly in the map (`:19`). Typing a bare integer line id was never a floor
operation.

### 6b. Server inference (`_resolve_run_line_id`, `main.py:18306`)

1. **Today's rule first:** exactly one `product_line_assignments` row for the item →
   that line. (Live: 18 batch → `granola`, 4 batch → `coconut`, 32 finished →
   `bulk_pack`, 9 finished → `pouch`.)
2. **New fallback by run type** when the item has zero or several assignments:
   `bake` → `granola` (1); `coconut` → `coconut` (2); `pack` → `pouch` (4) when
   `pack_format = 'bagged'`, `bulk_pack` (3) when `pack_format IN ('10lb','25lb')`,
   else NULL; `other` → NULL. Lines are looked up by `line_code`, not by id, so a
   re-seeded `production_lines` cannot mislabel a run.
3. Otherwise NULL.

Who the fallback catches, live: batch products 90008, 90025, 90026 (no assignment;
the two Kosher Ignition batches were noted at `SYSTEM_KNOWLEDGE.md:473`), 4 finished
with a `pack_format` but no assignment, and 11 finished with neither (those stay
NULL). The 63 existing assignments are reference data and are not edited.

`line_id` stays on the API as an optional override for `POST`/`PATCH` (tests at
`tests/test_production_runs.py:525-538` pin it; nothing on the dashboard sends it).
The column stays nullable; it is a board label, not a capacity input (Part 4 §2f).

---

## 7. Migration plan — `054_run_type.sql`

### 7a. Statements (053 conventions: no `BEGIN`/`COMMIT`, idempotent, catalog-checked, marker)

1. `ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS run_type text NOT NULL DEFAULT 'pack';`
   — backfills every existing row to `pack`, which is what every S1 row is.
2. `DO` block: add `production_runs_run_type_check CHECK (run_type IN
   ('bake','pack','coconut','other'))` if absent.
3. `ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS pan_yield_lb_used numeric(14,4);`
   plus `production_runs_pan_yield_lb_used_check CHECK (pan_yield_lb_used IS NULL OR
   pan_yield_lb_used > 0)` if absent.
4. Widen the unit vocabulary: `ALTER TABLE production_runs DROP CONSTRAINT IF EXISTS
   production_runs_planned_unit_check;` then re-add as `CHECK (planned_unit IS NULL OR
   planned_unit IN ('cases','lb','pans'))`. Drop-and-add is idempotent and the 053
   comment already says widening this list "is a migration, not a config edit".
5. Marker row `054_run_type` in `migration_markers`, `ON CONFLICT DO NOTHING`, with the
   053-style `RAISE NOTICE` including the row count and the count per `run_type`.

Not in 054: no change to `run_coverage`; no change to `products` (the yield exists,
§2b); no change to `production_lines` / `product_line_assignments`; no new index (the
list is filtered by `planned_date`, already indexed; add `(run_type, status)` only if
a board query needs it later). Apply with the session pooler on **port 5432**, never
6543, per `CLAUDE.md`; then `scripts/dump_prod_schema.sh` and a
`FACTORY_LEDGER_CHANGELOG.md` row.

### 7b. Do existing rows survive?

* **Production:** `production_runs` and `run_coverage` both hold **0 rows** (read-only
  check via port 5432 on 2026-09-15; markers 051–053 present, 053 applied 2026-09-14
  17:07 UTC). The backfill is a no-op. Any run created before 054 lands is a finished-SKU
  run and becomes `pack` with every other column already valid.
* **Test suite:** `tests/test_production_runs.py` creates its rows through the API per
  test; nothing is stored between runs. Changes needed there: the create helper sends
  `run_type` (or relies on the DB default for pack cases); the parametrised
  `({"ptype": "batch"}, "PRODUCT_NOT_SCHEDULABLE", 400)` case at `:495` becomes "batch
  product rejected for `pack`, accepted for `bake`"; new cases for pans conversion,
  `PAN_YIELD_REQUIRED`, routing-based coverage, and the line fallback. The five
  `EXPECTED_LOCK_SEQUENCE` rows (A26–A30) are unchanged because no lock changes.
* **`tests/schema/schema.sql`:** re-dumped after apply, as for 051–053.

### 7c. Order of operations

Migration applied by hand **before** the code that writes `run_type` merges (051/052/053
precedent, `FACTORY_LEDGER_CHANGELOG.md` rows 131/142/143). Between apply and deploy the
running S1 build keeps working: its `INSERT` omits `run_type` (default `pack`), omits
`pan_yield_lb_used` (nullable), and writes only `cases`/`lb` (still in the CHECK).

### 7d. Rollback

* **Code rollback** (redeploy the previous Railway build): safe at any time for the
  reasons in 7c. Bake runs created in between remain visible in the old list with
  `planned_unit = 'pans'` rendered as text and `recorded_qty` falling back to pounds;
  the old `PATCH` cannot re-convert them (it would hit `INVALID_UNIT`), which is
  acceptable for a short window.
* **Schema rollback** (`migrations/down/054_run_type_down.sql`, per the `046` down-file
  convention, wrapped in `BEGIN … COMMIT` because it is only ever run by hand):
  1. Refuse if any row has `run_type <> 'pack'` or `planned_unit = 'pans'` — print
     them and stop. Deleting those runs (and their `run_coverage` rows) is a data-loss
     step that needs its own explicit approval in the session; it is not folded into
     the down file.
  2. Restore `production_runs_planned_unit_check` to `('cases','lb')` — this is the
     safety: `ADD CONSTRAINT` fails if a `pans` row still exists.
  3. Drop `production_runs_run_type_check`, drop column `run_type`; drop the pan-yield
     check and column.
  4. Delete the `054_run_type` marker.
  `run_coverage` rows for pack runs are untouched by the rollback; rows for deleted
  bake runs go with them by explicit `DELETE`, never by cascade (there is no `ON DELETE
  CASCADE` on the FK and 054 does not add one).

### 7e. API contract delta (for the runs screen and tests; no yaml)

| Route | Delta |
|---|---|
| `POST /production/runs` | body gains **required** `run_type`; `planned_unit` admits `pans`; `line_id` optional as before. Response run object gains `run_type`, `pan_yield_lb_used`, `expected_lb` (bake/coconut), `uncovered_lb`. |
| `PATCH /production/runs/{id}` | `run_type` not editable (`422`); `pans` re-conversion; otherwise as built. |
| `GET /production/runs` | new optional filter `run_type`; each run carries the new fields. |
| `GET …/evidence`, `POST …/complete` | transaction-type filter by run type; `recorded_qty` in pans when applicable. |
| `PUT …/coverage` | routing-based product check for bake/coconut (§4b); codes unchanged. |
| `GET /products/search` | unchanged server-side; the screen stops filtering to `type = 'finished'` when the chosen run type is bake/coconut/other (`docs/design/runs-screen-spec.md` §"Implementation decisions" item 2 notes the filter is client-side). |

---

## Plain-English summary

Today a run can only be a finished product, so the floor cannot write down the thing it
actually plans first — "bake twelve pans of Classic #9 tomorrow". This amendment lets a
run be a bake (pans of a recipe), a coconut cycle, a pack (cases of a finished product,
which is what runs are today), or a one-off "other". A bake run's expected pounds are
computed from the recipe's existing per-pan weight, which the ledger already uses every
time a batch is recorded, so nothing new has to be maintained; the bulk granola it
produces is the batch product the system already tracks, so no new items are created.
Sales-order lines can be pointed at a bake run the same loose way they are pointed at
runs now — a promise, not a reservation — and whatever pounds are not promised show up
as "uncovered" on the run. Health already reads this without any change. Pack runs that
draw down that bulk are defined but their consumption link waits for S2, and the
amendment lists what must stay out of the schema so S2 has a clean start. The
production-line box comes off the form and is filled in from the item. The migration
adds two nullable-or-defaulted columns and one wider check, there are no rows in
production to convert, and a rollback of the code is safe at any point.

## Open questions for the owner

1. **Should the finished-SKU (`pack`) run stay creatable in this step?** It is what the
   Runs screen ships with today; keeping it means the floor can plan either level now.
   The alternative is to hide `pack` in the form until S2 wires WIP consumption.
2. ~~**Pan yield basis for coconut:** hydrated output (399.6 lb, what `/make` posts) or
   dry input (360 lb, the floor's number and the matrix dict's)?~~ **Resolved
   2026-09-15: 360 lb is the real pan weight.** Runs plan 360 per pan; `yield_multiplier`
   is not applied (§2a, §2d). See question 11 for the history it leaves behind.
3. **322.6 vs 323 lb for Classic #9** (and 380.12/380, 370.12/370): which is right, and
   should `products.default_batch_lb` be corrected? Not blocking; the run stores the
   value used.
4. **Whole pans only, or fractional?** `/make` counts whole batches; this draft accepts
   any positive number of pans and stores it as entered. Say if half-pans should be
   rejected.
5. **Add-ins at pack:** some SKUs gain pounds at pack (formula add-ins, `main.py:8532-8898`).
   This draft treats 1 lb WIP = 1 lb finished for coverage. Ignore in S1, or apply the
   add-in ratio where a BOM has one?
6. **Health wording:** should a shortage covered by a bake run read "covered by bake on
   Sep 16" rather than "covered by run on Sep 16"? Today's S2 sentence would be reused
   verbatim with no code change; a bake-specific word is a small S2 string change.
7. **Same line covered by both a bake and a pack run** before S2: allow with an on-screen
   warning (this draft), or block at `PUT …/coverage`?
8. **The three batch products with no line assignment** (90008, 90025, 90026): rely on
   the run-type fallback (this draft), or add the three `product_line_assignments` rows
   as a separate data fix?
9. **Keep `line_id` as an optional API override**, or remove it from the bodies so the
   inference is the only writer?
10. **Retire `_ORDERS_MATRIX_PAN_YIELD`** in favour of `default_batch_lb` in a later chore,
    so there is one pan number in the codebase?
11. **How to correct the historical coconut batches booked at 399.6 per pan** (§2d:
    131 sweetened makes since ~2026-02-17, 40,868 lb above the 360 basis, plus the
    forward fix). Choices, not exclusive: (a) **forward only** — set `yield_multiplier`
    to 1.0 on 90003/90004/90005 via `PUT /admin/products` so every make from that day
    posts 360 per pan, leave history as is, and let a physical count of the silos plus
    one `adjust` per product true up the balance; (b) **re-base history** — one
    `adjust` per product for the 11 % overstatement (−3,208 / −36,155 / −1,505 lb),
    which as §2d shows would push 90004 and 90005 negative and therefore cannot be
    done blind; (c) **amend the 131 makes** through the void/amend layer, which is the
    only route that also fixes `trace_event_lots` and the lot records but is 131
    corrections and changes numbers on every historical production report. Whichever
    is chosen, the two tests that pin `1.11 → 399.6` and the SYSTEM_KNOWLEDGE / CONTEXT
    passages need the same change, and the correction is its own PR with its own
    changelog row — not part of this amendment.

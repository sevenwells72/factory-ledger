# Sales-order state model — Phase A discovery findings

Branch: `feat/so-state-model` (off `origin/main` @ 750601e)
Date: 2026-09-10
Clone: `/Users/michaelgross/dev/factory-ledger`

**Status: Phase A complete. Finding A4 contradicted an `[ASSUMED]` premise in
the task spec; the owner ruled on it (see §Owner rulings) and Phase B proceeded
on those rulings.**

---

## A1 — Writers of `sales_orders.status` and `sales_order_lines.line_status`

| # | File:line | Function / endpoint | Column | Values written | Trigger |
|---|-----------|--------------------|--------|----------------|---------|
| 1 | `main.py:1800` | startup "Migration 007" block (in-app, runs on boot) | `sales_orders.status` | `'confirmed'` (only `WHERE status='new'`) | Process start; one-shot legacy sweep |
| 2 | `main.py:10456` | `_create_sales_order_core()` — the single INSERT path, shared by `POST /sales/orders` and the SO-intake approve flow | `sales_orders.status` | `'confirmed'` (hardcoded in the INSERT) | Manual order create; `POST /sales/orders/extract/approve` |
| 3 | `main.py:12275` | `update_order_status()` — `PATCH /sales/orders/{order_id}/status` | `sales_orders.status` | any of `new, confirmed, in_production, ready, invoiced, cancelled`, gated by `MANUAL_TRANSITIONS`; `shipped`/`partial_ship` explicitly rejected (400) | Dashboard (allowlisted) or Floor/office GPT (`updateOrderStatus` in openapi-gpt-v3.yaml) |
| 4 | `main.py:13006` | `ship_order_commit()` — `POST /sales/orders/{order_id}/ship/commit` | `sales_orders.status` | `'shipped'` if every physical line fully shipped, else `'partial_ship'` | Shipment commit |
| 5 | `main.py:12884` | `ship_order_commit()`, service-line branch | `sales_order_lines.line_status` | `'fulfilled'` | Shipment commit, `p.is_service = true` lines |
| 6 | `main.py:12947` | `ship_order_commit()`, physical-line branch | `sales_order_lines.line_status` | `'fulfilled'` / `'partial'` / `'pending'` from `quantity_shipped_lb` vs `quantity_lb` | Shipment commit |
| 7 | `main.py:12532` | `cancel_order_line()` — `PATCH /sales/orders/{order_id}/lines/{line_id}/cancel` | `sales_order_lines.line_status` | `'cancelled'` (guarded `line_status != 'fulfilled'`) | Line cancel (GPT op `cancelOrderLine`) |

Not a writer, but frequently mistaken for one: `main.py:12385`
(`PATCH /sales/orders/{order_id}` header edit) builds a dynamic `SET` clause
from `requested_ship_date / notes / notes_es / customer_id` only — `status`
appears solely in its `RETURNING` list. It *reads* status as a gate
(`ORDER_HEADER_LOCKED` unless status is `new` or `confirmed`).

**No DB trigger, rule, or view writes either column.** Every write is one of
the seven Python sites above.

### Consequences for the mirror
The spec's mirror (`closed→'shipped'`, `cancelled→'cancelled'`,
`reopen→'confirmed'`) writes values that sites 3 and 4 also write.
Two collisions to handle in Phase B:
* Site 3 (`PATCH .../status`) refuses `shipped`/`partial_ship` from a caller
  and enforces `MANUAL_TRANSITIONS`. The mirror must bypass that endpoint and
  write the column directly — it is not a transition the manual table allows
  (e.g. `confirmed → shipped` is not in `MANUAL_TRANSITIONS`).
* Reopen writes `'confirmed'` regardless of what the order's status was
  before it closed (e.g. an order that was `ready` and got short-closed
  reopens as `confirmed`, losing `ready`). This is per spec; flagged as a
  deliberate lossy step, not a bug.

---

## A2 — Readers that branch on status

### The computed `status=open` grouping
`GET /sales/orders` (`main.py:11363`, filter at `main.py:11410`):

```sql
if status == 'open':  query += " AND so.status NOT IN ('shipped','invoiced','cancelled')"
else:                 query += " AND so.status = %s"
```

The same predicate is recomputed per-row at `main.py:11433` as
`is_open = r['status'] not in ('shipped','invoiced','cancelled')`, and drives
`overdue`, plus three warning strings (no ship date / possible test order /
empty order). `overdue_only=true` uses the identical predicate
(`main.py:11420`).

**`partial_ship` and `new` are both "open"** under this definition.

### A second, different "open"
`GET /sales/orders/fulfillment-check` (`main.py:11552`) uses
`OPEN_STATUSES = ('confirmed','in_production','ready','partial_ship')` —
**excludes `new`**, and is a positive `IN` list rather than a negative one.
Two definitions of "open" already coexist in the backend.

### All status-branching readers

| File:line | Reader | Predicate |
|-----------|--------|-----------|
| `main.py:1449` | `_load_allocatable_line()` | `order_status IN ('cancelled','invoiced')` → `ORDER_NOT_ALLOCATABLE` |
| `main.py:5635` | `_customer_product_pool()` | `so.status <> 'cancelled' AND sol.line_status <> 'cancelled'` |
| `main.py:6856` | `check_open_orders_for_ship()` | `so.status NOT IN ('shipped','invoiced','cancelled')` |
| `main.py:10949` | production requirements demand | `so.status NOT IN ('shipped','invoiced','cancelled')` |
| `main.py:11032` | `SALES_ORDER_READINESS_SQL` | `sol.line_status <> 'cancelled'` (line level only; no order-status filter) |
| `main.py:11410/11420/11433` | `GET /sales/orders` | the `open` grouping above |
| `main.py:11514` | `POST /sales-orders/{so_number}/ready` | rejects `status IN ('shipped','invoiced','cancelled')` |
| `main.py:11589/11631` | `GET /sales/orders/fulfillment-check` | `OPEN_STATUSES` |
| `main.py:12337` | `PATCH /sales/orders/{order_id}` | `status NOT IN ('new','confirmed')` → `ORDER_HEADER_LOCKED` |
| `main.py:12261` | `PATCH .../status` | `MANUAL_TRANSITIONS[current]` |
| `main.py:13114` | packing-slip PDF | displays `so.status` |
| `main.py:13595` | `GET /sales/dashboard` | counts `WHERE status NOT IN ('invoiced','cancelled')` |
| `main.py:13613` | `GET /sales/dashboard` overdue list | `so.status NOT IN ('shipped','invoiced','cancelled')` |
| `main.py:14585` | global search | displays `so.status` |
| `main.py:15932` | `_load_demand()` (production planning) | selects `so.status`, filters downstream |

### Dashboard consumers (not to be changed, must keep working)
`dashboard/dashboard.js:2076-2080` hardcodes the same lists client-side:
```js
SALES_ORDER_OPEN_STATUSES   = ['new','confirmed','in_production','ready','partial_ship']
SALES_ORDER_CLOSED_STATUSES = ['shipped','invoiced','cancelled']
```
used at `:2246` (status filter), `:2499` (calendar), `:3501-3510` (line badges).
This is exactly the complement of the backend's `status=open`, so the mirror
keeps the dashboard correct with no dashboard edit.

### openapi-gpt-v3.yaml operations exposing/filtering status
| Line | Operation | Exposure |
|------|-----------|----------|
| 787 | `listOrders` (`GET /sales/orders`) | `status` query param, free-form `type: string` |
| 891 | `updateOrderStatus` (`PATCH /sales/orders/{order_id}/status`) | request body `OrderStatusUpdate` |
| 343 | `OrderStatusUpdate` schema | `enum: [new, confirmed, in_production, ready, partial_ship, shipped, invoiced, cancelled]` |
| 821 | `createOrder` response | prose description mentions `status` |
| 878 | `updateOrderHeader` response | prose description mentions `status` |
| 967 | `cancelOrderLine` response | `line_status` |

**The file currently holds exactly 30 operations — the hard ceiling in
CLAUDE.md.** Even if the spec permitted a schema change, there is no room for
a new op. Consistent with the spec's "no openapi-gpt-v3.yaml changes".

---

## A3 — CHECK constraint and migration numbering

`sales_orders` (from `tests/schema/schema.sql:1423`, a schema-only pg_dump of prod):
```sql
id                 integer NOT NULL          -- <- related_so_id must be integer
status             text DEFAULT 'new'::text NOT NULL
CONSTRAINT sales_orders_status_check CHECK ((status = ANY (ARRAY[
    'new','confirmed','in_production','ready',
    'shipped','partial_ship','invoiced','cancelled'])))
```
Note the column default is still `'new'` even though every code path writes
`'confirmed'`.

`sales_order_lines.line_status`: `CONSTRAINT sales_order_lines_line_status_check
CHECK (line_status = ANY (ARRAY['pending','partial','fulfilled','cancelled']))`.

**Migration naming/numbering:** `migrations/NNN_snake_case_name.sql`, three-digit
zero-padded, contiguous. Highest present is `050_sales_doc_intake.sql`.
**Next number: `051`.** Reversals live in `migrations/down/NNN_<name>_down.sql`
(only 046 has one); dry runs in `migrations/dry-runs/`. Migrations are applied
manually against Supabase — they are *not* auto-applied on deploy (the in-app
`Migration NNN` blocks in `main.py` startup are a separate, older mechanism and
are numbered independently).

---

## A4 — Actor attribution ⚠️ CONTRADICTS THE `[ASSUMED]` PREMISE

**Assumed:** "a surface-level actor can be derived from the key on every write path."
**Actual:** it cannot. There are two attribution mechanisms and neither satisfies this.

### Mechanism 1 — `_operator_id()` (`main.py:8204`), used by the SO write paths
```python
def _operator_id(auth_context: Any) -> str:
    """Phase-1 compatibility shim for the existing shared credential."""
    if isinstance(auth_context, dict) and auth_context.get("operator_id"): ...
    if isinstance(auth_context, str) and auth_context.strip(): ...
    return "legacy-shared-key"
```
The auth dependency it is fed is `_: bool = Depends(verify_api_key)`, and
`verify_api_key` → `_authorize_api_key` returns a bare **`True`** on every
success path (`main.py:2340-2350`). `True` is neither a dict nor a str, so
**`_operator_id(_)` returns the constant `"legacy-shared-key"` on 100% of
calls** — master key and dashboard key are indistinguishable downstream. This
is what the existing cancel path (`main.py:12291`) and ship path
(`main.py:12902`) record as `released_by`.

Worse, `"legacy-shared-key"` is **explicitly banned by the existing test suite**:
* `tests/test_sales_order_allocations.py:663` — asserts the string never appears
  in a migration file.
* `tests/test_expected_receipts.py:542` — asserts attribution columns are NULL
  rather than a placeholder.
* `tests/test_sales_order_allocations.py:216` — asserts the attribution column
  has no default, commented "never a `'legacy-shared-key'` placeholder default".

### Mechanism 2 — `caller_source_tag()` (`main.py:4743`), the sanctioned one
```python
def caller_source_tag(request, body_tag=None):
    """Interim attribution until FR-15 (user attribution) exists: a plain-text
    SOURCE tag, never a fake user id.
      * scoped dashboard key -> 'dashboard' (body ignored)
      * master key -> the caller-supplied tag if any, else NULL
    Deliberately NOT the 'legacy-shared-key' operator_id placeholder."""
```
This is what the **allocation** endpoints use — i.e. the closest analogue to
the new close/cancel/reopen endpoints, and the same subsystem as the release
path in A5 (`main.py:11920`, `main.py:12113`).

But it returns a **surface, not an actor**, and critically it returns **NULL on
master-key calls** unless the caller passes a body tag. The master key is
accepted on *every* route including allowlisted ones, so a close/cancel/reopen
can legitimately arrive with no derivable attribution at all.

### Why this blocks
The spec says `state_changed_by` comes "from the actor found in A4" and that
backfilled rows get `state_changed_by='migration-051'`. Those two together imply
a meaningful non-null identity string. The repo supports neither:
* using `_operator_id` writes a constant the test suite forbids;
* using `caller_source_tag` writes `'dashboard'` or **NULL**, and NULL will be
  the value for every master-key/GPT-originated call.

Real user attribution is tracked in this repo as unbuilt work — **FR-15**,
referenced in both mechanism docstrings.

### How `sales_order_flags` records "floor" (second half of A4)
Not from the key either. `migrations/037_sales_order_flags.sql` declares
`ready_by text DEFAULT 'floor'`, and `POST /sales-orders/{so_number}/ready`
(`main.py:11516`) sets it from a **client-supplied body field**:
```python
ready_by = (req.by or "floor").strip() or "floor"
```
So `'floor'` is a caller-asserted label with a hardcoded default — it is not
evidence that the key identifies a surface.

---

## A5 — Allocation/reservation release

**Single release path:** `_release_active_allocations()` (`main.py:896`), called
with exactly one of `order_id=` / `line_id=` / `allocation_id=`:

```sql
UPDATE sales_order_allocations
   SET status = 'released', released_at = clock_timestamp(),
       released_by = %s, release_reason = %s
 WHERE status = 'active' AND <scope>
RETURNING id, product_id, sales_order_line_id, lot_id, quantity_lb
```

**Release is audited in-row**, not in a separate log: `sales_order_allocations`
carries `released_at` / `released_by` / `release_reason`
(`migrations/044_sales_order_allocations.sql:55-57`), with
`CHECK ((status <> 'released') OR (released_at IS NOT NULL))` (line 63).

Existing callers and their `reason` values:
| Caller | Scope | reason | released_by |
|--------|-------|--------|-------------|
| `main.py:12292` — order cancel via `PATCH .../status` | `order_id` | `'order_cancelled'` | `_operator_id(_)` → `'legacy-shared-key'` |
| `main.py:12541` — `cancel_order_line()` | `line_id` | `'line_cancelled'` | `_operator_id(_)` → `'legacy-shared-key'` |
| `main.py:12118` — `POST .../allocations/{id}/release` | `allocation_id` | `'manual_release'` | `caller_source_tag(request)` |

The order-cancel path also runs `_lock_allocation_product()` then
`_expire_auto_fifo_allocations()` per distinct product **before** releasing —
the new close/cancel handlers must replicate that ordering or they will race
auto-FIFO expiry.

`_release_active_allocations` is a pure `UPDATE`: **no ledger postings, no
`trace_events`**. So the spec's "zero ledger rows / zero trace_events" holds
for free — trace emission is opt-in via explicit `emit_trace_event()` calls
(`main.py:2065`), never implicit.

---

## A6 — Effective quantities ✅ ASSUMPTION HOLDS

`SALES_ORDER_READINESS_SQL` (`main.py:11019`) already computes exactly what
Fulfillment and the `critical` health tier need, and already applies both
exclusions the spec asks for:

* **Cancelled lines excluded** — `line_base` CTE: `WHERE sol.line_status <> 'cancelled'` (`main.py:11032`).
* **Voided shipments excluded** — `posted` CTE: `WHERE ct.effective_status = 'posted'`, joined through `ledger_current_transactions` (`main.py:11045`); `shipped_eff` sums only those (`main.py:11062-11071`).
* Service lines (`p.is_service`) are held separate and excluded from the physical roll-up (`main.py:11288`).

`_line_readiness()` (`main.py:11178`) then yields per line:
`ordered_lb`, `shipped_recorded_lb`, `shipped_effective_lb`,
`remaining_lb = max(0, ordered - shipped_effective)`,
`shortage_lb = max(0, remaining - coverable)`, `allocated_lb`, `blockers[]`.
`_load_sales_order_readiness()` (`main.py:11277`) aggregates to the order.

Both `GET /sales/orders` and `GET /sales/orders/{order_id}` already call it, so
Fulfillment and Health can be derived with **no new SQL**:
* `fulfillment` = `unshipped` if `shipped_effective ≈ 0`; `shipped` if
  `remaining_effective ≈ 0`; else `partial` (all with `BALANCE_EPSILON`).
* `health.critical` ← the existing `shortage` blocker
  (`_blocker("shortage","block", f"Short {shortage:.4f} lb …")`, `main.py:11237`).
* `health.info` ← the existing `unallocated` blocker (`main.py:11239`) gated on
  `_allocations_enforced()` (`main.py:367`, env `ALLOCATIONS_ENFORCED`, default `false`).

A single-line variant, `_line_shipped_effective()` (`main.py:432`), applies the
same `effective_status='posted'` rule for one line.

**Route-ordering note for `GET /sales/orders/counts`:** it must be declared
*before* `GET /sales/orders/{order_id}` (`main.py:11711`) or FastAPI will match
`counts` as an `order_id` path param and `resolve_order_id` will 404 on it.
`GET /sales/orders/fulfillment-check` (`main.py:11544`) is the existing
precedent and is declared first.

---

## A7 — Running the tests in this clone

* `pytest.ini`: `testpaths = tests`, `addopts = -ra -q`, marker `db` for tests
  needing Postgres.
* `tests/conftest.py` hard-refuses Python ≥ 3.13 (`assert sys.version_info < (3, 13)`)
  because on 3.14 the suite exits 0 with zero tests collected. Pinned to 3.12.
* Tests run against `TEST_DATABASE_URL` **only**; if it points at a production
  host the session aborts, and if it is unset `DATABASE_URL` is scrubbed and
  db tests skip.
* `.venv-test/` is **absent in this clone** (present in the `Documents/factory-ledger`
  clone). Toolchain needed is available: `/opt/homebrew/bin/python3.12` ✅ and
  `/opt/homebrew/opt/postgresql@17/bin/psql` ✅.

Setup + run:
```bash
python3.12 -m venv .venv-test
.venv-test/bin/pip install -r requirements.txt -r tests/requirements-test.txt   # incl. pyyaml
scripts/setup_test_db.sh          # creates factory_ledger_test from tests/schema/schema.sql
scripts/run_tests.sh              # pinned interpreter + TEST_DATABASE_URL + expat workaround
```
`scripts/run_tests.sh` passes args through, so chunked foreground runs are
`scripts/run_tests.sh tests/test_x.py -q`.

New-schema caveat: `tests/schema/schema.sql` is a dump of **prod**, so it will
not contain migration 051's columns. Migration 051 must be applied to the test
DB (or replayed by the test setup) before the new tests can pass.

---

## Owner rulings (2026-09-10) — resolution of the A4 contradiction

Finding **A4** contradicted the `[ASSUMED]` premise *"a surface-level actor can
be derived from the key on every write path."* It cannot. The owner ruled as
follows, and Phase B was built on these rulings.

### 1. `state_changed_by`
Precedence, highest first:
1. an optional `changed_by` body field on `close` / `cancel` / `reopen`, stored
   **verbatim** when present;
2. otherwise `caller_source_tag(request)` — `'dashboard'` for the scoped key,
   a caller-supplied tag for the master key, else NULL;
3. NULL is allowed and expected.

`'legacy-shared-key'` is **never** written.

> **What the column means.** `state_changed_by` records a **surface or a
> self-reported identity — not an authenticated identity.** `'dashboard'` means
> the request presented the scoped dashboard key; any other non-null value was
> asserted by the caller and is not verified. Per-user attribution remains
> blocked on **FR-15**. Do not use this column as an audit control that has to
> withstand a hostile actor.

### 2. Mirror + `status_before_exit`
`sales_orders.status` is written **directly, inside the same transaction as the
state change**, bypassing `PATCH /sales/orders/{id}/status` and
`MANUAL_TRANSITIONS` entirely (A1 showed the manual table cannot express these
transitions). Migration 051 therefore also adds **`status_before_exit TEXT NULL`**:

* on `close` / `cancel`, the current `status` is copied into `status_before_exit`
  **before** the mirror overwrites it;
* on `reopen`, `status` is restored from `status_before_exit` (fallback
  `'confirmed'`) and `status_before_exit` is cleared.

This removes the lossy `ready → confirmed` reopen noted in §A1.

### 3. The two "open"s
`GET /sales/orders?status=open` and `fulfillment-check`'s positive `OPEN_STATUSES`
list are both **left unchanged**. Both collapse onto `state='open'` when
`status` is eventually retired; that is a post-deprecation cleanup, not this PR.

### 4. Health is provisional
The tier computation is isolated in a single function, `compute_so_health()`,
labelled **"v1 — provisional. Tier rules under review; response shape is the
contract."** Callers may depend on the shape of
`{level, reasons, info}`; they may not depend on the specific tier rules.

### 5. Counts endpoint key
The Factory-Ready bucket is keyed **`ready_to_ship`**, not `ready`, to avoid
colliding with the `ready` legacy *status* value and the `ready` flag on
`sales_order_flags`.

### 6. A6
Confirmed to hold. Fulfillment and Health are built on
`SALES_ORDER_READINESS_SQL`; **no new SQL was added for either.**

---

## Cross-review fix pass (2026-09-10) — policy rulings

The Codex cross-review of PR #40 found two blockers and eight should-fix items.
The owner's rulings below are policy, not preference, and the branch implements
them.

### State is authoritative

Any write that ships, allocates, or otherwise advances an order requires
`state='open'`, **checked under the order row lock**. Otherwise 409 naming the
current state and pointing at reopen (`ORDER_NOT_OPEN`, with
`suggested_action: "reopen"`).

Enforced at:
* `ship_order` commit — `main.py`, after the `FOR UPDATE OF so` read
* `ship_order` preview — read-only, no lock, but it must not promise something
  the commit will refuse
* `_load_allocatable_line()` — under the existing `FOR UPDATE OF so, sol`
* `PATCH /sales/orders/{id}/status` for the two exit-shaped values

### Lock ordering

**`sales_orders` row first (`SELECT … FOR UPDATE`), then product/lot locks.**
The allocation path already did this. `ship_order_commit` did not: it read the
order unlocked and then took product locks inside the ship plans, which is both
a deadlock shape against the exits and a lost-update window — a close could
commit between the eligibility read and the shipment write. `_load_so_for_state_change()`
was also strengthened from `FOR NO KEY UPDATE` to `FOR UPDATE` so every state
path takes the identical lock.

### Legacy-cancellation policy — `PATCH /sales/orders/{id}/status`

Two of the eight legacy status values are administrative exits wearing an
operational costume. Letting them write `status` alone would leave the two
models disagreeing — exactly the drift the mirror exists to prevent — and would
make the legacy endpoint the way around the cancel guard.

| value | behaviour |
|---|---|
| `new`, `confirmed`, `in_production`, `ready` | unchanged; **never touch state** |
| `cancelled` | routes through the same logic as `POST /cancel`: order lock → `fulfillment='unshipped'` guard (409 `ORDER_ALREADY_SHIPPED` pointing at close/`short_closed` if partial or shipped) → `state='cancelled'`, `state_reason='other'`, `state_note='via legacy status endpoint'` → reservations released → `status` mirrored to `'cancelled'` |
| `invoiced` | routes through close: `state='closed'`, reason `shipped_recorded` when effective remaining is 0 across all non-cancelled lines else `shipped_not_recorded`, `state_note='via legacy status endpoint (invoiced)'`, reservations released, and **`status` mirrored to `'invoiced'`, not close's usual `'shipped'`**, so this endpoint's own legacy callers still see the status they asked for |

The legacy response keeps every field it had (`order_id`, `order_number`,
`previous_status`, `status`, `allocations_released`, `message`); the exit
branches add `state`, `state_reason`, `state_note`, `state_changed_by` and
`attribution_note` on top. Operational transitions add nothing.

### Cutover reconciliation

An idempotent sweep runs at startup beside the Migration 007 sweep
(`main.py`, "Migration 051-reconcile"):

```sql
UPDATE sales_orders
   SET state = 'cancelled', state_reason = 'other',
       state_note = 'reconciled from legacy status at startup',
       state_changed_by = 'startup-reconcile',
       state_changed_at = clock_timestamp()
 WHERE status = 'cancelled' AND state = 'open'
```

**Rows with `status IN ('shipped','invoiced')` and `state='open'` are NOT
touched.** "Open · Shipped" is a legitimate, expected combination — everything
physically went out but nobody has administratively closed the order yet — and
making that visible is precisely why the state model exists. Auto-closing them
would erase the distinction on the first boot after deploy.

### Other rulings applied

* **Exit release is order-scoped.** The legacy cancel path calls
  `_expire_auto_fifo_allocations()` per product, which expires *other orders'*
  stale auto-FIFO rows as a side effect. Reasonable for an allocation endpoint;
  wrong for an administrative exit — closing one order must not release a
  different customer's reservation, and `reservations_released` must list
  exactly the rows the call changed. `_release_order_reservations()` no longer
  expires; the original function and its other callers are untouched.
* **`changed_by`** is stripped of whitespace and otherwise stored verbatim.
  Over 120 characters is **rejected with 400**, never truncated — silently
  storing a clipped identity would quietly corrupt the one field whose purpose
  is saying who did this. `released_by` on reservations released during an exit
  comes from `caller_source_tag(request)` and **never** from the body: the body
  is a claim about who asked for the exit, not a fact about which surface
  released stock.
* **`related_so_id`** must exist, must not be the order itself, and is accepted
  **only** for `duplicate`/`superseded` (400 `RELATED_SO_NOT_APPLICABLE`
  otherwise). Validated identically in preview and commit.
* **Health info** reports `unallocated_need_lb` whenever it exceeds zero, not
  only when nothing at all is allocated — 100 lb remaining with 40 lb allocated
  is 60 lb unallocated, and the old test hid exactly the partially-covered
  lines most worth seeing. Still info-only; never affects level.
* **`attribution_note` is returned in preview responses too**, alongside
  `resulting_state_changed_by`, so the caveat travels with every response that
  mentions attribution rather than only the committing one.
* **Terminal states cannot carry a NULL reason.** `state_reason IN (…)` is NULL
  when the column is NULL, and a CHECK constraint admits NULL — only an
  explicit `FALSE` rejects a row. Each terminal branch now carries
  `state_reason IS NOT NULL`, and the whole expression is wrapped in `IS TRUE`.
* **Backfill idempotency is a durable marker**, `migration_markers(name PK,
  applied_at)`, introduced in 051 because the repo had no applied-migration
  table. A guard derived from the order rows cannot tell "already backfilled"
  from "backfilled, and every row has since legitimately moved on".

---

## Follow-up: `_operator_id()` is a no-op placeholder

**Not fixed in this PR by owner ruling** — `_operator_id()` and `verify_api_key`
are untouched here.

`_operator_id()` (`main.py:8204`) returns the constant `"legacy-shared-key"` on
**100% of calls**, because `verify_api_key` → `_authorize_api_key` returns a bare
`True` and `True` is neither a dict nor a str. Every caller that passes the
auth dependency into it is therefore recording a placeholder, not an actor:

* `main.py:12292` — order cancel via `PATCH .../status` → `released_by`
* `main.py:12541` — `cancel_order_line()` → `released_by`
* `main.py:12902` — `ship_order_commit()` → `released_by` on the ship plan

Meanwhile `sales_order_allocations.released_by` on the *manual* release path
(`main.py:12118`) correctly carries `caller_source_tag(request)`. **The same
column therefore holds two incompatible kinds of value depending on which code
path released the row** — `'dashboard'`/NULL from manual releases,
`'legacy-shared-key'` from cancels and ships.

Suggested follow-up, in rough order of cost:
1. Change the three `_operator_id(_)` call sites to `caller_source_tag(request)`
   (each handler needs a `request: Request` parameter added). Cheap; makes
   `released_by` internally consistent immediately.
2. Then delete `_operator_id()` and its shim docstring.
3. FR-15 proper — real per-user attribution — supersedes all of the above and
   is the only thing that makes any of these columns trustworthy.

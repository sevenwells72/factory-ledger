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

**Exercised 2026-09-11:** the tier rules were rewritten (see § Health v2 below)
and the shape was not touched. The label is now **"v2 — time-aware. Shape is
the contract."** — the ruling stands, it is the version number that moved.

### 5. Counts endpoint key
The Readiness bucket — **Ready to Ship — the floor's flag, currently stored as
Factory Ready** (`sales_order_flags.ready`, migration 037) — is keyed
**`ready_to_ship`**, not `ready`, to avoid colliding with the `ready` legacy
*status* value and the `ready` flag on `sales_order_flags`. The key was the
first place the floor's own words for this dimension were used; as of design
standards v1.2 the label follows everywhere, STATUS-001 and STATUS-012
included, and `compute_so_health()` says `Not Ready to Ship` in its reasons.
The stored name is kept in the parenthetical, never dropped — a reader still
has to be able to find the column.

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
* `ship_order` commit — `main.py`, after the `FOR NO KEY UPDATE OF so` read
* `ship_order` preview — read-only, no lock, but it must not promise something
  the commit will refuse
* `_load_allocatable_line()` — under the existing `FOR NO KEY UPDATE OF so, sol`
* `add_order_lines()` — under the order row lock it now takes first
* `PATCH /sales/orders/{id}/status` for the two exit-shaped values

### Lock ordering

**`sales_orders` row first, then product/lot locks.** The allocation path
already did this. `ship_order_commit` did not: it read the order unlocked and
then took product locks inside the ship plans, which is both a deadlock shape
against the exits and a lost-update window — a close could commit between the
eligibility read and the shipment write.

> **Superseded in the second review pass.** This section originally specified
> `FOR UPDATE`, and briefly strengthened `_load_so_for_state_change()` to
> match. That was wrong: `FOR UPDATE` conflicts with the FK `KEY SHARE` locks
> and created two deadlock paths of its own. The canonical rules — strength,
> order, and every lock site — are **[The locking invariant](#the-locking-invariant-documented-not-incidental)**
> below. Read that, not this paragraph.

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

## The locking invariant (documented, not incidental)

### Scope of the deadlock-freedom claim

**This PR claims deadlock freedom for sales-order write paths only.** Those are
the paths that touch `sales_orders` / `sales_order_lines` and the reservations
hanging off them: the three administrative exits, `ship_order`, the allocation
create/release endpoints, the line mutations, `add_order_lines`, and the legacy
`PATCH /sales/orders/{id}/status`.

**Explicitly NOT covered, and not modified here:** the production and batch
commit paths — `make` (`main.py:7683`, `:7701`), `pack` (`:8165`) and
`reassign_lot` (`:9912`) — take `lots` locks under their own rules. `make` and
`pack` do order their multi-lot locks `ORDER BY id ASC`, and `reassign_lot`
locks with `FOR UPDATE OF l`, but none of them participates in the sales-order
lock order and none is exercised by the concurrency tests here. Whether they
agree with each other, and with the SO paths, on product/lot ordering is a
**separate follow-up PR with its own cross-review**. Nothing below should be
read as a claim about them.

### Ruling 1 — strength: `FOR NO KEY UPDATE`

**Every `sales_orders` and `sales_order_lines` row lock is `FOR NO KEY UPDATE`.
Not `FOR UPDATE`.**

Verified before adopting it:

* **Nothing deletes either table.** There is no `DELETE FROM sales_orders` or
  `DELETE FROM sales_order_lines` anywhere in the codebase.
* **Nothing rekeys either table.** The key columns are `sales_orders.id`,
  `sales_orders.order_number` (unique) and `sales_order_lines.id`. No `UPDATE`
  writes any of them; `order_number` is written only by the `BEFORE INSERT`
  trigger `trg_order_number`, and the two `BEFORE UPDATE` triggers touch only
  `updated_at` and the `created_at` guard.

So the weaker mode still mutually excludes all three writers — exit, ship,
allocate — because they all take it on the same row.

What it buys: `FOR UPDATE` conflicts with `FOR KEY SHARE`, which is what a
foreign-key check takes on the *referenced* row (see table **(c)**). Under
`FOR UPDATE`, an exit holding order A and writing `related_so_id = B` blocks
against anything holding B, and two reciprocal duplicate-cancellations (A→B and
B→A) deadlock outright: Postgres kills one with **40P01**.

### Ruling 2 — order

```
sales_orders row  ->  sales_order_lines rows  ->  product/lot rows
```

and **within products/lots, always ascending id**.

Ascending matters as much as the sequence. If each writer locked products in
whatever order its own lines happened to be in, shipping an order whose lines
read (P2, P1) would take P2 and wait for P1 while an exit on another order held
P1 and waited for P2 — both forever.

---

### (a) Explicit row locks acquired

| # | Function | Lock target | Strength | Step |
|---|---|---|---|---|
| A1 | `_lock_sales_order()` (`main.py:412`) | `sales_orders`, one row | `FOR NO KEY UPDATE` | 1 |
| A2 | `_lock_sales_order_lines()` (`:436`, `:444`) | `sales_order_lines`, order's rows or a named subset, `ORDER BY id` | `FOR NO KEY UPDATE` | 2 |
| A3 | `_lock_allocation_products()` (`:459`) | delegates to A4 once per product, **ascending product id** | — | 3 |
| A4 | `_lock_allocation_product()` (`:471`, `:477`) | `lots` `ORDER BY id`, then `sales_order_allocations` `ORDER BY id` | `FOR UPDATE` | 3, within one product |
| A5 | `_load_so_for_state_change()` (`:12171`) | `sales_orders`, one row | `FOR NO KEY UPDATE OF so` | 1 — head of every exit |
| A6 | `_load_allocatable_line()` (`:1508`) | `sales_orders` + `sales_order_lines`, one each | `FOR NO KEY UPDATE OF so, sol` | 1 then 2 — see note below |
| A7 | `ship_order` commit, order read (`:13765`) | `sales_orders`, one row | `FOR NO KEY UPDATE OF so` | 1 |
| A8 | `ship_order` commit, line lock (`:13775`) | `sales_order_lines`, all order rows | `FOR NO KEY UPDATE` | 2 |
| A9 | `ship_order` commit, product pre-lock (`:13823`) | every shipment product, ascending | via A3 | 3, before **either** planning loop |
| A10 | `add_order_lines()` (`:13403`) | `sales_orders`, one row | `FOR NO KEY UPDATE` | 1, before the eligibility read and the inserts |
| A11 | `cancel_order_line()` (`:13499`, `:13500`, `:13510`) | order row → that line → that product | `FOR NO KEY UPDATE`, then A4 | 1 → 2 → 3 |
| A12 | `update_order_line()` (`:13542`, `:13548`, `:13611`) | order row → that line → that product | `FOR NO KEY UPDATE`, then A4 | 1 → 2 → 3 |
| A13 | `update_order_line()` allocation shrink (`:13621`) | `sales_order_allocations`, that line's active rows | `FOR UPDATE` | 3, inside A4 |
| A14 | `_release_order_reservations()` (`:12267`) | the order's allocation products, ascending | via A3 | 3 |
| A15 | `create_sales_order_allocation()` (`:12830`) | via A6, then exactly one product via A4 | — | 1 → 2 → 3 |
| A16 | `release_sales_order_allocation()` (`:13023`) | one product via A4 — **no order row lock** | `FOR UPDATE` | 3 only (see note) |
| A17 | `update_order_status()` legacy PATCH | via A5, then A14 on the exit branches | — | 1, then 3 |
| A18 | `_consume_allocation_row()` (`:824`) | one `sales_order_allocations` row | `FOR UPDATE` | 3, inherited (see **(d)**) |
| A19 | `_upsert_live_allocation()` (`:1576`) | one `sales_order_allocations` row | `FOR UPDATE` | 3, inherited |
| A20 | `_shrink_overallocated_products()` (`:1416`, `:1437`) | `sales_order_allocations` | `FOR UPDATE OF soa` | 3, inherited |
| A21 | `_coalesce_lot_allocations()` (`:1454`, `:1464`) | `sales_order_allocations` | `FOR UPDATE` | 3, inherited |
| A22 | `_void_ship_allocations()` (`:1062`, `:1072`) | `sales_order_allocations` | `FOR UPDATE` | 3, inherited |
| A23 | `_prepare_restore_ship_allocations()` (`:1327`) | `sales_order_allocations` | `FOR UPDATE` | 3, inherited |
| A24 | `ship_order` **preview** | *no lock* — read-only | — | n/a; the `state='open'` check still runs |
| A25 | `set_sales_order_ready_flag()` (`main.py`) | `sales_orders`, one row (`SELECT ... state FROM sales_orders FOR NO KEY UPDATE`) | `FOR NO KEY UPDATE` | 1 only; writes `sales_order_flags` (no FK to `sales_orders`); no ordering edge added |

Locks outside the sales-order graph, listed so the table is exhaustive rather
than because they interact: `find_open_expected_receipt` (`:5036`, optional),
`settle_expected_receipt` (`:5059`), `update_expected_receipt` (`:5325`),
`approve_extracted_receipts` (`:6454`), `approve_extracted_sales_order`
(`:6605`), `update_supply_request` (`:6985`), `_append_transaction_correction`
(`:8403`), `_append_transaction_line_correction` (`:8536`),
`correct_certification` (`:9313`), `verify_product` (`:10293`), and the
out-of-scope `make` (`:7683`, `:7701`), `pack` (`:8165`) and `reassign_lot`
(`:9912`).

**Note on A6 — `_load_allocatable_line`.** It selects
`FROM sales_orders so JOIN sales_order_lines sol JOIN products p` and locks
`FOR NO KEY UPDATE **OF so, sol**`. The `FROM` clause order is not what makes
this correct — a join's row-mark order is not guaranteed to follow the textual
`FROM` list, and `products` is deliberately *not* marked at all. What makes it
correct is the explicit `OF so, sol`: Postgres marks the listed relations in
the order given, so this single statement takes step 1 then step 2 and never
touches a product row.

**Note on A16 — `release_sales_order_allocation`.** It reads the order
unlocked and then takes only the product lock, so it acquires step-3 locks
without step 1. That is safe rather than sloppy: it never reaches *backwards*
for an order or line lock afterwards, so it cannot invert against a 1→2→3
writer. It is also not state-gated — releasing a reservation on an order that
has since closed is harmless, and the exits release the same rows anyway.

---

### (b) Implicit DML locks

Every `INSERT`/`UPDATE`/`DELETE` takes `ROW EXCLUSIVE` on the table and an
implicit row lock on each row it writes, without any `FOR …` clause. These are
the ones that matter for ordering, all inside a transaction that already holds
the explicit locks above:

| # | Site | Rows implicitly locked |
|---|---|---|
| B1 | `_apply_state_change()` | the `sales_orders` row (already held via A5) |
| B2 | mirror write of `sales_orders.status` | same row as B1, same statement |
| B3 | `ship_order` commit — `UPDATE sales_order_lines … quantity_shipped_lb`, `… line_status` | line rows (already held via A8) |
| B4 | `ship_order` commit — final `UPDATE sales_orders SET status` | order row (already held via A7) |
| B5 | `cancel_order_line()` — `UPDATE sales_order_lines … 'cancelled'` | the line (already held via A11) |
| B6 | `update_order_line()` — `UPDATE sales_order_lines SET …` | the line (already held via A12) |
| B7 | `add_order_lines()` — `INSERT INTO sales_order_lines` | new rows; takes `KEY SHARE` on the parent order, see C1 |
| B8 | `_release_active_allocations()` / `_shrink_active_allocations()` | `sales_order_allocations` rows (inside A4/A14) |
| B9 | `_expire_auto_fifo_allocations()` | `sales_order_allocations` rows (inside A4) |
| B10 | startup reconcile sweep | every `sales_orders` row matching `status='cancelled' AND state='open'` |

B4 is the one worth remembering: an endpoint that skipped the A7 row lock would
still end up blocked here, at the very end of its transaction. That is exactly
why the race tests assert on the *waiting statement* and not merely on "is it
blocked".

---

### (c) FK `KEY SHARE` sites

Every insert into a referencing table takes `FOR KEY SHARE` on the referenced
row. These are what `FOR UPDATE` would have conflicted with.

| # | Referencing write | Takes `KEY SHARE` on |
|---|---|---|
| C1 | `INSERT INTO sales_order_lines` | `sales_orders` (parent), `products` |
| C2 | `INSERT INTO sales_order_allocations` | `sales_orders`, `sales_order_lines`, `products`, `lots` |
| C3 | `INSERT INTO sales_order_shipments` | `sales_order_lines`, `transactions` |
| C4 | `INSERT INTO shipment_lines` | `sales_order_lines`, `shipments`, `transactions`, `products` |
| C5 | `INSERT INTO shipments` | `sales_orders`, `customers` |
| C6 | `INSERT INTO trace_events` | `sales_orders`, `transactions`, `customers` |
| C7 | `INSERT INTO sales_order_allocation_reactivations` | `sales_order_lines`, `transactions`, `ledger_corrections` |
| C8 | **`UPDATE sales_orders SET related_so_id = …`** | the *other* `sales_orders` row |

C8 is the deadlock that motivated Ruling 1: it is the only site where a writer
holding one order row reaches for a *second* order row, and under `FOR UPDATE`
two of them pointing at each other deadlock.

---

### (d) Inherited prelocks — functions that rely on a caller's lock

These take no order/line lock of their own and are only correct because their
caller already holds one. Calling any of them from a new path without first
taking the caller's locks would break the invariant silently.

| # | Function | Requires the caller to hold |
|---|---|---|
| D1 | `validate_lot_deduction()` | the lot row, `FOR UPDATE` — stated in its own docstring |
| D2 | `_sales_order_ship_plan()` | the product's lots (A4/A9) when `lock=False`; takes them itself when `lock=True` |
| D3 | `_consume_sales_order_allocations()` | the product lock (A4) |
| D4 | `_expire_auto_fifo_allocations()` | the product lock (A4) |
| D5 | `_release_active_allocations()` | the product lock (A4) for the products involved, and — for exits — the order row (A5) |
| D6 | `_shrink_active_allocations()` | the product lock (A4) and the allocation rows (A13) |
| D7 | `_release_order_reservations()` | the order row (A5); it takes the product locks itself via A3 |
| D8 | `_validate_allocation_addition()` | the line (A6) and product (A4) locks |
| D9 | `_load_sales_order_readiness()` | nothing — read-only, deliberately takes no locks |

> Row 15 of the previous version of this table described
> `_release_active_allocations()` / `_shrink_active_allocations()` as if they
> established their own position in the lock order. **They do not.** They are
> inherited-prelock functions (D5, D6): they lock only `sales_order_allocations`
> rows, and they sit at "step 3" only because of what their caller already
> holds. The corrected wording is in D5/D6 above.
>
> **And "every caller holds step 1" is not true.** `release_sales_order_allocation()`
> (A16) calls `_release_active_allocations()` holding **only** the product
> lock — it reads the order unlocked and never takes the order row. So D5's
> requirement is caller-dependent, not universal:
>
> * the exits and the legacy PATCH reach it holding the order row (A5) **and**
>   the product lock (A3/A4);
> * `cancel_order_line()` reaches it holding the order row, the line, and the
>   product (A11);
> * `release_sales_order_allocation()` reaches it holding the product lock
>   alone (A16).
>
> The last case is safe because it never reaches *backwards* for an order or
> line lock afterwards, so it cannot invert against a 1 → 2 → 3 writer — but it
> is a genuine exception to the pattern, not an instance of it, and anything
> that added an order-row or line lock to that path *after* the product lock
> would create an inversion.

---

### (e) Offline and test-only operations

| # | Operation | Locks taken |
|---|---|---|
| E1 | Migration 051 — `ALTER TABLE sales_orders ADD COLUMN …` | `ACCESS EXCLUSIVE` on `sales_orders` |
| E2 | Migration 051 — `ADD CONSTRAINT` (in the `DO` block) | `ACCESS EXCLUSIVE` on `sales_orders` |
| E3 | Migration 051 — `CREATE INDEX idx_sales_orders_state` | `SHARE` on `sales_orders` (blocks writes; non-concurrent by design, this is an offline apply) |
| E4 | Migration 051 — backfill `UPDATE`s | `ROW EXCLUSIVE` + row locks on every matched order |
| E5 | Migration 051 — `CREATE TABLE migration_markers` | new relation, no contention |
| E6 | Startup reconcile sweep | `ROW EXCLUSIVE` on `sales_orders` (B10) |
| E7 | Test teardown (`_cleanup_order`) | implicit row locks on the order's child rows, then the order |
| E8 | Test harness holder connections | `FOR NO KEY UPDATE` / `FOR SHARE` on an order row, or `FOR UPDATE` on a product's lots — deliberately parked to force interleavings |

E1–E3 are why migration 051 must be applied out of band, before the code that
reads the new columns deploys: they take `ACCESS EXCLUSIVE` and will queue
behind any long-running transaction.

---

### What the concurrency tests actually prove

`tests/test_sales_order_state_model.py` holds two families, and they prove
different things.

**`TestStateRaces`** — one writer against a holder parked on the order row.
Each asserts, via `pg_blocking_pids()` against a backend PID captured with
`pg_backend_pid()` *before* the endpoint runs, that the writer is blocked and
that **the statement it is blocked on is the initial order-row `SELECT`**.
Mutation-verified: removing the row lock from `ship_order`, from
`_load_allocatable_line` or from `add_order_lines`, with the state check left
in place, fails the assertion.

**`TestNoDeadlock`** — two writers staged into a specific interleaving by
parked holders, then released.

Being precise about what this does and does not establish, because the
distinction matters:

* **What a holder does.** It drives each writer to a known step and stops it
  there, so the two are provably overlapping and holding partial lock sets
  when they are released. It removes the schedule-dependence that made the
  earlier version of these tests pass vacuously — one writer simply finishing
  before the other started.
* **What the chain assertion establishes.** Each test asserts the *shape* of
  the resulting blocking graph, not merely that both writers are blocked
  somewhere. That shape is what differs between the current lock order and the
  pre-fix one, which is why these tests fail under the mutations rather than
  surviving them.
* **What a passing test does NOT establish.** That the pre-fix code would have
  deadlocked on this schedule is *not* proved by the test passing — a test
  that passes says only that the current code completes. The counter-evidence
  comes from the mutation table in the PR body: each mutation restores one
  piece of the pre-fix locking and the corresponding test then fails, twice of
  the three with Postgres reporting SQLSTATE **40P01** outright.
* **What none of them establish.** Absence of deadlock in general. These are
  four specific schedules over sales-order write paths. They are not a proof
  of deadlock freedom across all interleavings, and they say nothing at all
  about `make` / `pack` / `reassign_lot`.

> Postgres reports only the **direct** blocker. When two writers queue on the
> same rows, the first names the holder and the second names the first, so
> reachability checks follow the blocking chain; where a test needs one
> specific link — "the exit is blocked *by the reduction*" — it pins that
> direct edge instead.

**These are transaction-body concurrency tests, not request-identical.** They
call the endpoint *functions* directly rather than issuing HTTP requests, so
they exercise each endpoint's real lock sequence but not the full request path
— no routing, no auth dependency, no response envelope, no middleware. That is
deliberate: one `TestClient` serialises both threads through a single ASGI
portal, so there would be no concurrency left to test, and two `TestClient`s
each run the app's startup and shutdown against the same module-level
`db_pool` and race to close it. The trade-off is that a defect living in the
request layer rather than the transaction body would not be caught here.

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

### Sales Orders list follow-ups

* `overdue_only` path in `list_sales_orders` has no SQL `LIMIT` and
  `_so_derived_fields` scans all `line_readiness_by_line` per order —
  O(orders × lines). Fix: group `line_readiness_by_line` by `sales_order_id`
  once in `_load_sales_order_readiness`. Fine at current volume.
* Sales Orders list: legacy Filter/Hide-ready-to-ship/Sort/Resize bar is
  redundant with tabs; remove Hide ready to ship, move Sort/Resize into table
  header, keep customer filter (Step 4 or 3.1).
* Pallets column renders '—' for all retail SKUs (7.5 lb, 2.63 lb cases) —
  cases-per-pallet exists only as a hardcoded 10 lb/25 lb table at
  `dashboard/pallet-calculations.js:8-11`. Fix: migration 052
  `products.cases_per_pallet`, expose in `pallet_lines`, dashboard prefers
  payload then table then '—'. Needs Blubber's retail ti-hi numbers.

---

## Health v2.1 — factory-local dates, a warning window, reason suppression (2026-09-10)

Branch: `feat/so-health-v2-1` (off `origin/main` @ 7613af3, which carries v2).
Code, tests and docs only — no migration, no `dashboard/` change.
`compute_so_health()` is still the single place tiers are decided, still fed by
`SALES_ORDER_READINESS_SQL`, and still issues no SQL of its own.

### Why v1 was replaced (v2)

v1 made **any** stock shortage critical. The same 500 lb short is a different
fact at three days out than at three months, and painting both red meant the
red said nothing — the board read as a list of orders rather than a queue of
work. v2 made the ship date half of every reason and half of every tier.

### What v2.1 changes

Three things, each fixing a way v2 still cried wolf or got the day wrong:

1. **`today` is the factory's date, not the server's.** The API runs UTC, so
   from 20:00 Eastern onward `date.today()` had already rolled over and an
   order due *today* was reported as *1 day overdue* — on the shift most likely
   to be looking at the board. Every date-vs-today comparison in the health and
   counts paths now goes through `_factory_today()`.
2. **A shortage past the warning window is `info`, not `warning`.** On a
   make-to-order book, an unproduced order with a month of runway is not a
   problem — it is the normal state of the work. v2 spent a warning badge on
   every one of them, which taught the board to be ignored a second time.
3. **The "Not Ready to Ship" reason yields to a shortage.** An order that
   cannot ship for want of material is not also news for want of a flag, and
   two reasons on one order read as two problems.

### The clock

`FACTORY_TZ` — the timezone the factory's calendar is kept in.

* **Default `America/New_York`.** Read on **every call**, like the day windows,
  so a move or a mistake is fixable without a deploy.
* An unknown zone (`KeyError`/`ZoneInfoNotFoundError`) or a malformed one
  (`ValueError`) falls back to the default rather than raising. Same rule as the
  windows: this is a read path, and a bad env var must not 500 the board.
* Deliberately **separate from module-level `PLANT_TIMEZONE`**, which the
  ledger, shipment and production paths read. Same value today; retuning the
  sales-order health clock must not silently retune theirs.

What it governs, and nothing else:

| Site | Before | After |
|---|---|---|
| `compute_so_health()` default `today` | `date.today()` (UTC on Railway) | `_factory_today()` |
| `GET /sales/orders/counts` → `overdue` bucket | `date.today()` | `_factory_today()` |
| `GET /sales/orders` → each order's `overdue` field | `date.today()` | `_factory_today()` |
| `GET /sales/orders?overdue_only=true` filter | SQL `CURRENT_DATE` (server clock) | parameterised `_factory_today()` |

That last row was its own latent bug: a SQL `CURRENT_DATE` filter and a Python
`date.today()` field could disagree inside a single response, so at 20:00 ET
`?overdue_only=true` would return an order the same payload labelled
`"overdue": false`.

**Out of scope, deliberately unchanged:** the expected-receipts overdue flag
(already on `get_plant_now()`), `GET /sales/dashboard`'s overdue/due-this-week
lists, the production planner's demand horizon, and every ledger, shipment and
allocation path.

### The tiers

Evaluated on **open orders only**. Highest tier wins; every applicable reason is
listed; `today` is always the factory's date.

| Level | Condition | Example reason |
|---|---|---|
| `critical` | Stock shortage on any non-cancelled line **and** `ship_by <= today + SO_HEALTH_CRITICAL_DAYS` (a past `ship_by` is inside that window by definition) | `Short 735 lb on 2 lines — ships in 3 days` |
| `warning` | Stock shortage with `ship_by <= today + SO_HEALTH_WARNING_DAYS` but outside the critical window, **or** with no ship date at all | `Short 500 lb — ships in 8 days` |
| `warning` | Overdue (`ship_by < today`, `fulfillment != 'shipped'`) with **no** shortage | `12 days overdue — stock on hand` |
| `warning` | Ready to Ship not set, `ship_by <= today + 2`, and **nothing short** | `Not Ready to Ship — ships tomorrow` |
| `info` | Stock shortage with `ship_by` further out than `SO_HEALTH_WARNING_DAYS` — stated, never tiered | `Short 1400 lb — ships in 13 days` |
| `info` | `unallocated_need_lb > 0` on a line — reported, **never** affects the level | `60 lb not allocated on SKU-1 (line #12)` |
| `quiet` | Nothing above applies | — |
| `quiet` | The order is `closed` or `cancelled` — off the board, so no reasons and no info at all | — |

### The knobs

Both are day counts read from the environment on **every call**, the same way
`ALLOCATIONS_ENFORCED` and `FACTORY_READY_REQUIRED` are, so either window can be
retuned without a deploy. Anything that is not a non-negative integer — a typo,
a float, a negative — falls back to its default rather than raising.

| Env var | Default | Question it answers |
|---|---|---|
| `SO_HEALTH_CRITICAL_DAYS` | 5 | How close before a shortage stops being a thing to plan around and becomes a thing to fix today? |
| `SO_HEALTH_WARNING_DAYS` | 10 | How close before it is worth mentioning as a tier at all? Beyond this it is `info`. |
| `FACTORY_TZ` | `America/New_York` | Which calendar are all of those days counted on? |

They are independent: moving one never moves the other. The **Ready to Ship**
window is a separate in-code constant (`SO_HEALTH_READY_DAYS`, 2), deliberately
*not* env-tunable and *not* tied to either day window — the floor's flag is a
same-week concern, not a supply one.

If the two day windows are ever set inverted (`critical > warning`), the level
still cannot outrun its reasons: `critical` is gated on the shortage having
earned a place in `reasons`, so a misconfiguration produces a quiet order with
an `info` line rather than a red badge with nothing to act on.

### Reason wording

Each reason is a whole instruction — how short, over how many lines, and by
when — because a bare `Short` sends the operator to open the order. The time
half comes from one helper, `_so_ship_phrase()`: `ships in 14 days` ·
`ships tomorrow` · `ships today` · `12 days overdue`, and nothing at all when
the order carries no ship date.

Three deliberate details:

* Shortages are **aggregated to one reason**, not one per line (v1 emitted one
  each). Ten per-line reasons bury both numbers the operator needs. `on N
  lines` is omitted when N is 1.
* Crossing the warning window changes **which list** the shortage line lands in,
  not what it says: the `info` wording is character-for-character the `reasons`
  wording, `on N lines` rule included.
* Two reasons are suppressed rather than stacked, both on the same `not
  short_lines` guard and for the same reason — when there *is* a shortage its
  reason already carries the date, and a second line reads as a second problem:
  * **overdue-with-stock-on-hand**, suppressed since v2;
  * **Not Ready to Ship**, suppressed as of v2.1. With the stock on hand the
    flag *is* what stands in the way, so there it is kept. Suppressed means
    gone, not demoted — it does not reappear in `info`.

### What did not change

The response shape `{level, reasons, info}` — owner ruling 4 — is untouched, and
so is every caller. Overdue-with-stock-on-hand is still `warning`. Not-Ready
within two days is still `warning`. A shortage with no ship date is still
`warning` — no deadline means nothing to be inside of, and it stays visible
either way. `closed`/`cancelled` are still silent. The `info` tier still reports
unallocated pounds whether or not `ALLOCATIONS_ENFORCED` is set.

### Tests

`tests/test_sales_order_state_model.py`: 192 tests in the file, 40 of them new.

* **Tier matrix** — 29 rows driven straight against `compute_so_health()` with a
  pinned `today`, covering every row of the table above and **both sides of all
  three boundaries**: `today + 5` critical vs `today + 6` warning, `today + 10`
  warning vs `today + 11` info, `today + 2` Not-Ready warning vs `today + 3`
  quiet. Rows may now assert expected `info` as well as `reasons`.
* **The clock** — pins an *instant* rather than a date, via a `_freeze()` helper
  that stands the process up as a UTC server (`datetime.now(tz)` honours the
  zone it is handed; `date.today()` returns the UTC day). An order due Sep 10
  evaluated at 20:00 Eastern is **not** overdue and reads `ships today`; the
  same order at 00:30 Eastern the next morning is `1 day overdue — stock on
  hand`. Both day windows are also proven to count from the factory date, and
  `FACTORY_TZ` is proven to move the calendar (one instant, `America/New_York`
  and `America/Los_Angeles`, two different "today"s) and to fall back on
  garbage.
* **Suppression** — stated as a pair so the rule cannot be half-reverted:
  identical inputs apart from the shortage, and the flag reason appears in
  exactly one of them. Plus an end-to-end pair through the endpoint.
* **Windows as env vars** — each widened, narrowed, defaulted, fed garbage, and
  proven independent of the other; plus the inverted-pair guard.
* **Counts and list on the same clock** — the `overdue` bucket, an order's
  health badge, the list's `overdue` field and the `?overdue_only=true` filter
  are all checked at both frozen instants, including a test whose whole point is
  that the bucket and the badge agree.

Mutation-verified, all four changes:

| Mutation | Result |
|---|---|
| `_factory_today()` → `date.today()` in `compute_so_health()` | **5 fail**, incl. `test_an_order_due_today_is_not_overdue_at_8pm_eastern`. The 00:30 test still passes — at that instant both clocks agree, which is what makes it the control. |
| `_factory_today()` → `date.today()` in the counts endpoint | **2 fail**, both counts-clock tests |
| drop the `not short_lines` suppression guard | **6 fail**, 3 matrix rows + 3 suppression tests |
| warning window `<=` → `<` | **2 fail**, `shortage-exactly-the-warning-boundary` and the narrowed-env-var test |

The v2 mutation check on the critical window still holds: `<=` → `<` fails
`shortage-exactly-the-boundary` and the narrowed-env-var test, and nothing else.
Full suite: **798 passed** (758 before this branch).

---

## Step 2d — Health strings: aggregated info, one number format, product names

The tiers were right and the sentences were not. Three defects, all in wording,
none in the level:

1. **`info` was one entry per line.** A twelve-line order with partial
   allocations produced twelve near-identical sentences, and the total — the
   number the operator is actually asking for — appeared in none of them.
2. **Numbers were raw.** `{unallocated:g}` and `{total_short:g}` gave
   `1400 lb` and `13500 lb`: no separator past a thousand, a decimal point
   whenever the pounds were not round. STATUS-006 exists because re-reading a
   quantity is how the wrong quantity gets shipped.
3. **Lines were named by bare SKU.** `on SKU-1` is a lookup the reader has to
   do at the moment they are trying to decide something.

The shape stays a contract. `{level, reasons, info}` is unchanged and gains one
key.

### `info` aggregation

One entry per order, not per line — the same rule the shortage reason has had
since v2, and for the same reason:

| Lines with unallocated pounds | Entry |
|---|---|
| 5 | `24,000 lb not allocated across 5 lines` |
| 2 | `1,200 lb not allocated across 2 lines` |
| 1 | `60 lb not allocated on Granola SS Chocolate Chip (70003)` |
| 0 | — no entry |

`across N lines` is omitted when N is 1: there is no count worth stating and
nothing to disambiguate, so the single line names itself instead — which is
also the only place a line identity still appears in a health string. Lines
whose unallocated pounds fall under `BALANCE_EPSILON` are not lines: a rounding
crumb must not inflate the count, nor turn a one-line order into a two-line one.

When `ALLOCATIONS_ENFORCED` is off, ` (allocations not enforced)` is appended
**once**, to that one entry — STATUS-011, a caveat is stated once at first
relevance; repeated down a list it stops being read, including the time it
mattered. Enforcement changes the wording and nothing else: the pounds are
unallocated either way, the level never moves, and `info_detail` is identical
with the flag on and off. That is asserted as a pair so the suffix cannot drift
into carrying meaning.

### `health.info_detail` — the rows behind the sentence

Aggregation does not lose the per-line facts, it moves them. `info_detail` is a
list the popover expands the sentence back into, one entry per line that has
unallocated pounds:

```json
"info_detail": [
  {"line_id": 412, "sku": "70003",
   "product_name": "Granola SS Chocolate Chip", "unallocated_lb": 10000.0},
  {"line_id": 413, "sku": "70011",
   "product_name": "Granola Maple Pecan", "unallocated_lb": 14000.0}
]
```

Three properties callers may rely on:

* **Always present.** `[]` when there is nothing to expand — including on a
  `closed` or `cancelled` order, which stays silent in every field. A caller
  reads the key without checking whether it is there.
* **Pounds are raw, not formatted.** `unallocated_lb` is a number, because a
  string cannot be summed, sorted or re-rounded. Its entries sum to exactly the
  pounds in the sentence; the front end runs them through its own STATUS-006
  formatter. Formatting belongs to the layer that renders, and `info_detail`
  does not render.
* **`sku` and `product_name` are separate fields**, not the joined label. The
  popover may want them in a column each.

### One number formatter — STATUS-006

`_fmt_number()` is the only thing in this module that turns a number into text.
Thousands separators always, pounds to whole numbers, never a trailing decimal:
`13,500 lb`, not `13500.0000`, not `13500 lb`, not `13,500.00 lb`.

**Half rounds up, via `Decimal(ROUND_HALF_UP)` — not Python's own formatting.**
`f"{607.5:,.0f}"` is `608` but `f"{606.5:,.0f}"` is `606`: `format` rounds half
to *even*, so the same shortage prints larger or smaller depending on the digit
before it. A shortage that prints smaller than it is reads as less urgent than
it is.

Counts go through the same function — `on 2 lines`, `across 5 lines`,
`1,200 days overdue`. Not because a line count needs a separator today, but so
that nothing in a health string is hand-formatted and there is no second place
for a format to drift. Day counts are formatted in `_so_ship_phrase()`, pounds
in the shortage reason and the unallocated entry.

| Input | Output | Why it is in the table |
|---|---|---|
| `999` | `999` | below the separator |
| `1000` | `1,000` | the separator boundary |
| `999.5` | `1,000` | rounds up *and* gains a separator |
| `607.5` | `608` | half rounds up |
| `606.5` | `607` | the control — `format` gives `606` here |
| `12345.4` | `12,345` | rounds down |
| `12345.5` | `12,346` | rounds up |
| `13500.0000` | `13,500` | the stored representation, formatted |
| `None` | `0` | never renders blank |

### `product name (SKU)`

`_so_line_label()` renders `Granola SS Chocolate Chip (70003)` wherever a line
is named in a reason or an info string. A bare SKU is a lookup; a bare name does
not say which pack size.

**It costs no query.** `SALES_ORDER_READINESS_SQL` already selects `p.name AS
product` and `p.odoo_code AS sku` on every line row, and
`_load_sales_order_readiness()` already carries both into the dict health reads
— the name was being fetched and discarded. There is no per-line round trip and
no second query.

It degrades rather than printing an artefact: name only when there is no code,
code only when there is no name, and the literal `line` when there is neither —
never `Granola Maple ()` and never a bare `()`.

### What did not change

`{level, reasons, info}` and every rule about which facts land in which tier.
The `info` shortage wording is still character-for-character the `reasons`
wording. Unallocated pounds are still reported whether or not allocations are
enforced, and still never escalate. `closed`/`cancelled` still report nothing at
all.

### Tests

`tests/test_sales_order_state_model.py`: **231 tests in the file, 39 of them
new**; full suite **837 passed** (798 before this branch).

* **Aggregation** — one line, two lines, five lines, and a sub-epsilon line
  that must not be counted; plus an end-to-end pair through the endpoint (one
  partially-allocated line, and two on one order).
* **The enforcement suffix** — on and off, stated as a pair whose only
  difference is the suffix, plus a proof that it never raises the level and
  never touches `info_detail`.
* **`info_detail`** — one row per line, rows sum to the sentence, always present
  and `[]` when empty, unaffected by the flag.
* **Formatting** — a thirteen-row boundary table on `_fmt_number()` including
  `606.5` as the half-even control, plus the STATUS-006 audit itself run against
  the assembled strings: no three-decimal number, no value of 1,000 or more
  without a separator.
* **Product names** — a seven-row table on `_so_line_label()` covering both
  halves, either half missing, whitespace, and neither; plus the substitution
  end-to-end, where the name and code come from the readiness query's own
  columns.

Mutation-verified, every new rule:

| Mutation | Result |
|---|---|
| `ROUND_HALF_UP` → default (half-even) | **2 fail**, `_fmt_number[606.5-607]` and `[0.5-1]` — the 607.5 row still passes, which is what makes it the control |
| drop the thousands separator | **16 fail** |
| `_so_line_label()` returns the bare SKU | **7 fail** |
| un-aggregate: one `info` entry per line | **6 fail** |
| drop the not-enforced note | **3 fail** |
| `across N lines` boundary `> 1` → `>= 1` | **6 fail** |
| drop `info_detail` from the returned shape | **10 fail** |

---

## FR-15 attribution — per-actor keys (step 5a, migration 052, 2026-09-11)

This is the follow-up the section above calls "FR-15 proper", delivered
narrowly: the **backend** half. It gives the attribution columns something
truthful to hold. It does not yet change what the dashboard sends, so in
production nothing is attributed differently until the Codex UI step ships.

### What was added

`actors(id, name, role, key_hash, active, created_at, last_used_at)` —
migration 052, one row per human who writes through the ledger. `role` is
`owner | floor | office`, CHECK-constrained. `key_hash` is
`sha256(plaintext)`; the plaintext is never stored, so a database dump is not
a set of credentials. Keys are minted out of band by
`scripts/mint_actor_keys.py`, which prints each plaintext once and writes
hash-only INSERT SQL for the Supabase SQL editor. There are no seed rows.

### Resolution order

`_authorize_api_key()` is the single choke point for every authenticated
route, and its order is load-bearing:

1. **No key** → 401 `API key required`. Unchanged.
2. **`API_KEY`** (master) → authorized everywhere, `actor = None`,
   `key_kind = 'legacy_ledger'`. Unchanged.
3. **`DASHBOARD_API_KEY`** (scoped) → authorized only on
   `DASHBOARD_KEY_ALLOWLIST`, `actor = None`,
   `key_kind = 'legacy_dashboard'`. Unchanged.
4. **`sha256(key)` in `actors` where `active`** → `actor` attached to
   `request.state`, `key_kind = 'actor'`, authorized on exactly
   `DASHBOARD_KEY_ALLOWLIST`.
5. **Anything else** → the historical rejection, verbatim: 403 on the header
   dependency, 401 on the packing-slip query-param one.

Steps 1–3 return before step 4 runs. That is the whole legacy-key policy in
one sentence: **a call made with either legacy key never reaches the actor
code at all**, so it cannot be affected by the table's contents, its absence,
or a failed load. `caller_source_tag()` and `_state_changed_by()` read
`request.state.actor`, which is `None` for those calls, and fall into exactly
the branches they had before — `'dashboard'` for the scoped key, the body tag
or NULL for the master key.

A deactivated actor resolves to nothing, because the cache query filters on
`active`. Its key is then rejected identically to a key that was never
minted — which is what deactivation is meant to mean.

**On the rejection status code.** FR-15's brief asked for "unknown key → 401
exactly as today". Those two clauses disagree: today an unknown key is **403**
on the header dependency (`tests/test_dashboard_api_key.py::
test_missing_key_401_and_wrong_key_403` pins it) and 401 only on the
packing-slip query-param path. "Exactly as today" won, because the merge
criterion was that every existing caller keeps working unchanged. Changing it
is a one-line edit to `verify_api_key`'s `invalid_status` and a separate
decision.

### Scope of an actor key

Exactly `DASHBOARD_KEY_ALLOWLIST` — the same routes the shared dashboard key
reaches, no more. An actor key **replaces** that key; it does not upgrade it.
Handing a named person master-key reach (`/make`, `/adjust`, `/void`,
`/admin/*`, `POST /sales/orders`) would be a privilege escalation this step
has no mandate for. `role` is recorded and returned by `/auth/whoami` but
gates nothing yet; per-role scoping is a later decision, not an oversight.

### Caching

The active actor set is small and is cached whole for **60 seconds**
(`ACTOR_CACHE_TTL_S`), so authentication costs a dict lookup and no query. Two
consequences, both deliberate and both tested:

* a key minted less than a TTL ago may be rejected until the cache turns
  over — mint, wait a minute, then hand it out;
* a key deactivated less than a TTL ago keeps working for up to that long.
  **Deactivation is not an incident-response control.** If a key must die
  immediately, rotate `API_KEY`/`DASHBOARD_API_KEY` and restart the instance.

An **unknown** key never forces a refresh. Letting an unauthenticated caller
trigger a database round-trip per request is a free denial-of-service lever,
and the 60-second staleness is the price of not having one.

`last_used_at` is stamped at most once per key per 10 minutes
(`ACTOR_LAST_USED_THROTTLE_S`), best-effort: the column answers "is this key
still in use", which does not need per-request resolution, and a failed UPDATE
must never turn a valid request into a 500.

A failed load — most plausibly migration 052 not yet applied on this
database — caches **empty** for a full TTL and logs once, rather than raising.
With no actor resolvable, every key falls through to precisely the pre-FR-15
behaviour.

### Attribution, path by path

`caller_source_tag()` and `_state_changed_by()` each gained one branch at the
top: if an actor is resolved, return its name. Every sales-order write path
already routed its attribution through one of those two, so the following pick
it up without a handler change:

| Path | Column | Legacy key (unchanged) | Actor key |
|---|---|---|---|
| `POST .../close` | `sales_orders.state_changed_by` + released rows' `released_by` | `'dashboard'` / NULL | actor name |
| `POST .../cancel` | same | `'dashboard'` / NULL | actor name |
| `POST .../reopen` | `state_changed_by` | `'dashboard'` / NULL | actor name |
| `PATCH .../status` → cancelled/invoiced | same | `'dashboard'` / NULL | actor name |
| `POST .../allocations` | `sales_order_allocations.created_by` | `'dashboard'` / NULL | actor name |
| `POST .../allocations/{id}/release` | `released_by` | `'dashboard'` / NULL | actor name |
| `POST .../ship/commit` | `released_by` on expired auto-FIFO rows | `'dashboard'` / NULL | actor name |
| `PATCH .../lines/{id}/cancel` | `released_by` | NULL (master-only route) | — |
| `PATCH .../lines/{id}/update` | `released_by` | **was `'legacy-shared-key'`** | actor name |
| `POST /sales-orders/{so}/ready` | `sales_order_flags.ready_by` | body `by`, default `'floor'` | actor name |

Two paths are deliberately absent:

* **`POST .../lines`** (add lines) writes no attribution at all, because
  `sales_order_lines` has no attribution column. Adding one is a schema change
  this migration has no mandate for.
* **`PATCH /sales/orders/{id}`** (header update) likewise writes none.

### The `_operator_id` follow-up, finished

The section above lists three call sites to fix and says the manual release
path was already correct. It missed a fourth: `update_order_line()` wrote
`released_by` from the placeholder at **both** its expire and its shrink site,
and was overlooked precisely because it had no `request: Request` parameter to
route through the shared helper. It has one now. `sales_order_allocations.
released_by` finally holds one vocabulary on every path that writes it.

`_operator_id()` itself stays: other subsystems (void, adjust, certifications,
ledger corrections) still call it, and removing it is a separate change.

### The actor outranks a self-reported identity

`changed_by` on the exit request bodies is still **accepted** — the GPTs send
it and hold the master key, which resolves no actor — but when an actor is
resolved, the actor wins. A self-reported identity must not override an
authenticated one, or anyone holding a personal key could sign someone else's
name to an exit. The over-length `CHANGED_BY_TOO_LONG` 400 still fires first,
for every key kind, so an over-long value is never silently discarded.

### Locks

**No lock was added, removed, or reordered by any of this.** The change is to
the values passed into already-existing attribution parameters, plus one
dependency that runs before any handler opens a transaction and uses its own
short-lived connection. `test_so_write_paths_take_the_same_locks_in_the_same_
order` (in `tests/test_sales_order_state_model.py`, appended to the existing
concurrency harness) freezes the ordered lock sequence of all eleven
sales-order write paths; the expected lists were generated from the source at
72b5546 and verified byte-identical afterwards.

### `GET /auth/whoami`

Returns `{"actor": {"name", "role"} | null, "key_kind":
"legacy_dashboard" | "legacy_ledger" | "actor"}`. Allowlisted, so both the
dashboard key and actor keys reach it. **Deliberately not in
`openapi-gpt-v3.yaml`** — that file is at its hard 30-operation ceiling and no
GPT needs this route.

### Deliberately deferred

* **The dashboard still sends `dashboard-key-2026`** (`dashboard/dashboard.js`
  line 2067, injected by `fetchSalesAPI`). Nothing in `dashboard/` was touched
  by this step, so **production attribution stays unattributed until the Codex
  UI step ships**. This PR makes per-user attribution possible; it does not
  make it happen.
* **Per-user GPT keys are a follow-up.** The office and floor GPTs hold
  `API_KEY` and would each need their own actor key, which means deciding
  whether a GPT is an actor at all or a surface acting for one. Until then the
  GPTs' writes keep recording their self-reported `created_by` tag.
* **Per-role authorization.** `role` is stored and reported, gates nothing.
* **Revocation latency.** Bounded by the cache TTL, as above.

# A7 — Order tickets (`feat/order-tickets`)

Built from `origin/main` @ `2e9432b` (PR #92) in worktree `~/dev/fl-a7-wt`.
Design: `docs/design/phase1-safe-operating-system.md` §1 (ticket lifecycle),
§1.3 (order prepares), §1.5 (duplicate warning), §4.3 (matrix), §10 row A7.

## What it is

Prepare → commit for every order action in the §4.3 matrix, with the A1
guarantees unchanged (single use, 10/30-minute expiry, bound to the preparing
identity and the exact payload, commit re-validates, replay returns the same
receipt, concurrent double-commit → one result, possible-duplicate warning).
Logic lives in **`order_tickets.py`**; shared files are touched only at named
hook points (listed below). Inputs are ids from `POST /resolve`; no name is
resolved inside a write.

| Ticket `action` | Prepare route | Roles (§4.3) | Live core reused |
|---|---|---|---|
| `create_order` | `POST /sales/orders/prepare` | owner, office, master | `_create_sales_order_core` (+ `_check_order_po`) |
| `add_order_lines` | `POST /sales/orders/{id}/lines/prepare` | owner, office, master | `_insert_order_lines_core` (the create core's line loop, extracted) |
| `update_order_line` | `POST …/{id}/lines/{line_id}/update/prepare` | owner, office, master | `_update_order_line_core` |
| `cancel_order_line` | `POST …/{id}/lines/{line_id}/cancel/prepare` | owner, office, master | `_cancel_order_line_core` |
| `update_order_header` | `POST …/{id}/header/prepare` | owner, office, master | `_update_order_header_core` |
| `update_order_status` | `POST …/{id}/status/prepare` | owner, office, master (`cancelled`/`invoiced` also need the cancel/close rows) | `_update_order_status_core` + `_validate_requested_order_status` |
| `mark_order_ready` | `POST …/{id}/ready/prepare` | owner, **floor**, office, master | `_set_sales_order_ready_flag_core` |
| `cancel_order` | `POST …/{id}/cancel/prepare` | owner, office, master | `_cancel_sales_order_core` |
| `close_order` | `POST …/{id}/close/prepare` | owner, office, master; reason `shipped_not_recorded` → **owner** (+ master) only | `_close_sales_order_core` |
| `reopen_order` | `POST …/{id}/reopen/prepare` | owner, master | `_reopen_sales_order_core` |

`{id}` accepts the numeric id or the `SO-…` number (same `resolve_order_id`
dependency as the direct routes). Commit is the existing
`POST /tickets/{ticket}/commit`. Receipts are `ORD-YYMMDD-NNN` (see
*Deviations*). The dashboard key reaches none of the ten prepares (A2 rule:
shared keys get nothing new); named actor keys and the master key reach all ten.

### How a prepare builds its draft

The prepare **dry-runs the live core inside a savepoint** (`SAVEPOINT
order_draft` … `ROLLBACK TO SAVEPOINT`), captures the response as the draft,
and only then inserts the ticket row through `write_tickets.issue_ticket()`.
So the draft is exactly what commit will return, the same validation errors
become draft blockers, and nothing the core wrote survives the prepare. Ids
the dry run consumed (`order_id`, `line_id`) are stripped from the draft and
the SO number is shown as `order_number_provisional` — all three are assigned
at commit. Side effects of the dry run: identity sequence gaps on
`sales_orders`/`sales_order_lines` (harmless) and the core's `logger.info`
line (fires on prepare too).

Commit runs the same core for real after the shared lifecycle checks in
`write_tickets.commit_ticket` (identity, payload hash, replay, matrix row on
the stored action, expiry, warning acknowledgements), then the module's own
`require_reason_permissions` (403 — ticket stays `prepared`), the actor-active
check, draft blockers, `validate(lock=True)`, receipt allocation and the post.
Any `HTTPException` < 500 inside the savepoint rolls the post back, marks the
ticket `rejected` and answers `409 TICKET_STALE` with the blocker — exactly
the A1 path. `state_changed` compares the order/lines snapshot taken at
prepare with the one at commit.

### Order rules kept (all from main.py)

- **customer_po duplicate block with explicit override.** Prepare computes
  the duplicates (`_dedupe_existing_sales_orders`) and issues
  `DUPLICATE_CUSTOMER_PO` with `requires_ack: true` (or `false` when the
  payload already carries `allow_duplicate_po: true`). Commit passes
  `allow_duplicate_po or 'DUPLICATE_CUSTOMER_PO' in acknowledged_warnings`
  into `_check_order_po`; a duplicate that appears *after* prepare makes the
  ticket stale (`TICKET_STALE` / `DUPLICATE_CUSTOMER_PO`). Same on a header
  edit that changes the PO or the customer.
- **No PO allowed, flagged.** `NO_PO` warning (`requires_ack: false`),
  `customer_po_status: "No PO"` in draft and receipt. Blank PO = no PO.
- **Service lines** (Pallet Charge etc.): zero pounds, priced per `each`,
  `is_service: true` — the PR #67 contract path (`save_contract=True`), so
  ticket-created and ticket-added lines carry `ordered_quantity`,
  `ordered_unit`, `ordered_case_weight_lb` and `amount`.
- **external_order_reference.** A client-supplied reference is kept; if the
  customer already has it, prepare returns blocker
  `EXTERNAL_ORDER_REFERENCE_EXISTS` naming the order. Without one, the
  receipt number is stored as the order's `external_order_reference` (design
  §1.3: the ticket *is* the external reference for chat). The direct route's
  PR #67 replay table is not written by tickets; its unique constraints still
  refuse a later direct create with the same reference.
- **State model.** Cancel/close/reopen reuse the same reason validation,
  `ORDER_NOT_OPEN` / `ORDER_ALREADY_SHIPPED` guards, reservation release and
  `state_changed_by = actor name` as the direct routes. Status changes keep
  `MANUAL_TRANSITIONS`; `shipped`/`partial_ship` are refused (draft blocker).
- **Possible duplicate (§1.5).** `create_order` and `add_order_lines`: a
  committed ticket for the same customer/order with the same lines in the
  last 24 h, by any actor → `POSSIBLE_DUPLICATE` (`requires_ack: true`).
- **`happened_at` for orders is the plant day** (orders carry an
  `order_date`, not an event time; never back-dated). Two identical prepares
  on the same day therefore hash alike and the earlier one is superseded.

### A6 named hook — ship gate

`order_tickets.shipment_gate(api, cur, order, close_reason)` is called for
`close_order` with `shipped_recorded` and for `update_order_status` →
`invoiced`. **Today it returns `[]`** (no gate — the existing close behaviour).
A6 replaces the body; both callers already treat the result as draft
blockers at prepare and as a stale-ticket refusal at commit.
`close_order_shipped_not_recorded` stays owner-only (+ master key), checked at
prepare and again at commit.

## Hook points in shared files

- `write_tickets.py`: `ORDER_ACTIONS` + `PREFIXES` (`ORD`), `ORDER_PREPARE_ROUTES`
  unioned into `ACTOR_ROUTES` (not `DASHBOARD_ROUTES`); `issue_ticket()`
  extracted from the `prepare` closure (same row shape, used by both);
  `commit_ticket` delegates to `order_tickets.commit` after the shared checks;
  `receipt_detail` adds `order`; `register_routes` calls
  `order_tickets.register_routes` at the end.
- `permissions.py`: new matrix row `close_order` (owner, office, master — the
  ticket action; `shipped_not_recorded` additionally needs the existing
  owner-only row) and EN/ES labels for the order actions.
- `main.py`: `_insert_order_lines_core` (the per-line loop of
  `_create_sales_order_core`, moved verbatim), `_validate_requested_order_status`
  (the two pre-connection 400s of `PATCH …/status`), and eight
  `_<handler>_core(cur, request, …)` extractions — handler bodies moved
  verbatim; the routes keep their own connection/exception shells and call the
  core. No behaviour change on any direct route (suite + source guards).
- Tests that pin handler source (`test_sales_order_state_model` lock order,
  `test_actor_attribution` placeholder guard, `test_released_by_attribution`)
  now inspect the handler **and** its core.

## Migration 067 — staging applied, prod only with owner approval

`migrations/067_order_tickets.sql` (down: `migrations/down/067_order_tickets_down.sql`):

- `write_tickets.action` CHECK gains the ten order actions (062's rebuild
  pattern: the existing definition is kept and extended);
- `sales_orders.ticket_id`, `sales_order_lines.ticket_id` — nullable FK to the
  creating ticket, two partial indexes;
- `write_tickets_order_idx` on `(result_ref->>'order_id')` for "which receipts
  touched order X" (edits are UPDATEs; they are found through `result_ref` and
  the existing `actor_write_audit` rows);
- marker `067_order_tickets`. Additive, rerunnable, no backfill, no trigger or
  view change. ADD COLUMN of a nullable column is catalog-only.

**Applied to STAGING 2026-10-09 00:50:28Z** (wrapper: `BEGIN; SET LOCAL
lock_timeout='5s'; SET LOCAL statement_timeout='30s'; SET LOCAL
search_path=public; \i …; COMMIT;` on the session pooler, port 5432, as the
table owner). Staging already carried 066 and 068; the constraint rebuild
preserved them. **Not applied to production.**

**Deploy order:** 067 must be applied **before** this code goes live on an
environment — the commit path inserts `write_tickets` rows with the new
actions and writes the two `ticket_id` columns. Staging: done; let Railway
build the merge. Production: apply 067 by hand (same wrapper), then merge.
Rollback: revert the merge → wait for Railway → optionally the down file,
which refuses while order tickets exist (committed tickets are immutable).

`tests/schema/schema.sql` ends with a pending `\ir ../../migrations/067_order_tickets.sql`
block — remove after the prod apply + `scripts/dump_prod_schema.sh`
(housekeeping PR, as for 058/060/061/065). Migration number 067 was the free
slot between 066 (`codex/supplier-label-backfill`, on hold) and 068
(`feat/fl-assistant`).

## Staging evidence

`scripts/check_order_tickets_staging.py --output docs/deployments/a7-staging-receipt.json`
ran the branch's real HTTP routes locally against the guarded staging database
(TestClient without lifespan: no startup migrations or sweeps; the URI is read
from the protected secret file and never printed). Result
(`docs/deployments/a7-staging-receipt.json`): synthetic customer `1000000002`,
order **SO-261009-001** created through ticket receipt **ORD-261008-001** by a
temporary office actor (`client_source=fl_assistant`), replay returned the same
receipt, a second identical prepare got `DUPLICATE_CUSTOMER_PO` +
`POSSIBLE_DUPLICATE` and its commit was refused without acknowledgement, add
lines → `ORD-261008-002`, floor actor marked ready → `ORD-261008-003`, floor
create and office `shipped_not_recorded` close refused 403. Both temporary
actors deactivated afterwards. (Order numbers come from the DB's
`CURRENT_DATE` in UTC — the pre-existing `generate_order_number` trigger —
while receipts use the plant day, hence `SO-2610**09**` vs `ORD-2610**08**`
for an evening entry.)

## Tests

`tests/test_order_tickets.py` (27): dry-run prepare + single-use commit +
replay + receipt page for create; names/self-reported identity rejected;
unknown customer/product/inactive customer as blockers; duplicate PO
(ack / override / appears-after-prepare → stale); No-PO flag; service line;
external reference kept / refused / direct-route conflict; possible duplicate
on create and add-lines; add/update/cancel lines; header edits (PO duplicate,
customer change, clear PO, locked after advance); status table + blocked
values + legacy cancel via status; `invoiced` without recorded shipment
owner-only; floor marks ready and is denied the other nine; cancel/close/reopen
per matrix + master key; role re-check at commit for the close reason; wrong
user / payload mismatch / unknown ticket / unused lot confirmations /
supersession / expiry; stale after close + `state_changed`; role change and
deactivation between prepare and commit; dashboard key denied on all ten
routes; direct routes unchanged (master create, office PATCH, status, ready,
close, floor 403); matrix rows; the concurrent double-commit race on a
dedicated database (one order, one receipt) plus the populated 067 down
refusing. Shared tests updated: `test_write_tickets` (067 in the isolated
database + up/down/up, route placeholder), `test_roles_a2` (`close_order` row),
`test_named_actor_writes` (actor scope contract), the three source guards.

Fresh local DB from `tests/schema/schema.sql` (+ pending 067): Python 2190
collected, all passed; Node 69/69. (A second full run on the *same* DB fails
`test_seed_staging::test_seed_is_idempotent_and_uses_no_known_actor_key`
because the pre-existing `RACE FG` / `LOCKORDER FG` race fixtures in
`test_sales_order_state_model` commit products ≥ 1e9 and never remove them —
unrelated to A7; rebuild the DB between full runs, as the other lanes do.)

## Deviations from the design text (for review)

1. **Receipt prefix `ORD`, not `SO`.** Order *numbers* are already
   `SO-YYMMDD-NNN`; a receipt box that accepts both would collide
   (`SO-261008-001` the order ≠ `SO-261008-001` the receipt). One-line change
   in `write_tickets.PREFIXES` if the owner prefers `SO`.
2. **`close_order` added to `ROLE_PERMISSIONS`** (same roles as
   `cancel_order`), so the ticket layer can enforce on the stored action;
   the direct `/close` route still gates as `cancel_order` + in-handler owner
   check, unchanged.
3. **Inactive customers are refused** (`CUSTOMER_INACTIVE`); the direct
   route only checks existence. Ids come from `/resolve`, which returns
   active customers.
4. **Expected-receipt prepare** (in the design's A7 row) is not in this PR's
   scope.
5. **`sales_order_create_receipts`** is not written by ticket creates; the
   ticket's own replay covers retries.

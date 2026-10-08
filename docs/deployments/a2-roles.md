# A2 — roles enforced in FL, entered_by on every write, back-dating limits

Built on `origin/main` `1f3f098` (design rev 3.7) in worktree `~/dev/fl-a2-wt`,
branch `feat/roles`. Design: `docs/design/phase1-safe-operating-system.md`
§4 (identity and roles, §4.3 matrix, §4.4 floor identity), §6 (`happened_at` /
`entered_at` / `entered_by`, §6.3 back-dating), §11 item 6 (D6).

## What A2 adds

**`permissions.py` (new module, no DB state).** `ROLE_PERMISSIONS[action] →
roles` is the §4.3 matrix as code, keyed on `write_tickets.action` where a
ticket exists. Columns: the three `actors.role` values plus the two shared keys
(`legacy_ledger` = master `API_KEY`, `legacy_dashboard` = `DASHBOARD_API_KEY`),
which keep exactly what they reach today and get nothing new. `require(action,
identity)` raises the one structured denial:

```json
403 {"error_code": "ROLE_NOT_ALLOWED", "message": "Luz (office) is not allowed to record a make.",
     "message_es": "…", "action": "make", "role": "office", "actor": "Luz", "key_kind": "actor"}
```

`permissions_for(identity)` is the `{action: bool}` map `GET /auth/whoami`
now returns (display only — FL enforces). Later chunks reuse the same calls:
A3b `/exceptions/*` (`list_exceptions`, `resolve_exception`, `approve_exception`),
A4 part 2 `/aliases` writes (`manage_aliases`), A6/A7 ship and order tickets,
A11 PIN sessions (a session resolves to an actor and lands in that role column),
F1 (greys out from whoami).

**Enforcement points (keyed on the ticket action, never the route):**

1. `POST /{action}/prepare` — `permissions.require(action, identity)` first,
   then `require_backdating(...)`; both 403 before any `write_tickets` row exists.
2. `POST /tickets/{t}/commit` — after the identity binding and the replay
   return, `SELECT role, active FROM actors … FOR SHARE` (review fix 2: the
   row lock is taken here and held to the end of the posting transaction, so a
   role change or deactivation cannot land between the check and the post — a
   concurrent `UPDATE actors` waits), then `require(row['action'], identity)`
   with that role (not the ≤ 60 s auth cache), then `require_backdating` against
   the commit clock. A denied commit leaves the ticket `prepared` (it expires
   normally) and posts nothing.

**`entered_by` on every write (migration 065).** `transactions.entered_by_actor_id`
and `ledger_corrections.entered_by_actor_id` (nullable FK `actors(id)`, partial
index on transactions). Every `INSERT INTO transactions` in `main.py` (receive,
make, pack, adjust, found, standalone ship, order ship) and both
`ledger_corrections` writers (void, records correction) now pass
`_entered_by_actor_id(request)`: the actor's id for actor keys, `NULL` for the
two shared keys — whose `operator_id` stays `'legacy-shared-key'` exactly as
before. The three entry fields on a transaction are therefore:

| Field | Column | Set by |
|---|---|---|
| `happened_at` | `occurred_at` | the user (defaults to now), plant time |
| `entered_at` | `created_at` | DB clock, immutable |
| `entered_by` | `entered_by_actor_id` (+ `operator_id` name snapshot) | `_authorize_api_key`, never the body |

Ticket commit responses and receipts carry `entry_timing`
`{status, hours_late, days_late, happened_at, entered_at, entered_by: {id,name,role,key_kind},
late_entry_exception_id}`; the draft carries the same block without
`entered_at`/`entered_by`.

**Back-dating (§6.3, D6) — real elapsed time, displayed in America/New_York.**
Review fix 3: every comparison (`permissions.timing`, `validate_inventory_occurred_at`,
the draft's `happened_vs_now_minutes`) is made on UTC instants via
`permissions.elapsed()`. Two aware datetimes that share the plant `ZoneInfo`
subtract and compare by *wall clock* in Python, which is an hour off across a
DST change (and prepare and commit could disagree, because the stored
`occurred_at` comes back with a fixed offset). Plant time is used only for
display, receipts and business-day math.

| `now − happened_at` | floor / office / shared keys | owner (and master key) |
|---|---|---|
| future > 5 min | draft blocker `OCCURRED_AT_IN_FUTURE` (unchanged) | same |
| 0 – 48 h | normal | normal |
| 48 h – 14 d | allowed; draft warning `LATE_ENTRY` (`requires_ack: false`, "This will be recorded as a late entry (N days)"); commit opens `exceptions(kind='LATE_ENTRY', severity='warn', owner_actor_id = active owner)` in the same transaction as the post | allowed, no exception |
| > 14 d | `403 BACKFILL_OWNER_ONLY {action:'backdate_over_14d', role, actor, days_late}` at prepare and again at commit | existing path: `backfill:true` required (`OCCURRED_AT_BACKFILL_REQUIRED` blocker otherwise), `created_at_source='api_backfill'`, `entry_backfilled=true`, no exception |

Boundaries are strict: 48 h 00 m is normal, anything beyond is late; 14 d 00 m
is late, anything beyond is owner-only. The classification is re-run at commit,
so a ticket prepared at 47 h and committed after the line opens the exception,
and one prepared at 13 d and committed after 14 d is denied.

**Direct (legacy) routes — the same matrix for named actors (review fix 1).**
Until A10 closes them, an office actor key reaches `POST /make` exactly as it
reaches `/make/prepare`, so the gate is route-keyed as well:
`permissions.ROUTE_ACTIONS[(METHOD, route)] → action` is applied in
`_authorize_api_key` (where design §4.3 puts it) for `key_kind='actor'` only,
giving the same `403 ROLE_NOT_ALLOWED` before the body is read or the handler
runs. `POST /sales/orders/{id}/close` is gated as an order edit (`cancel_order`
column) and the handler adds the owner-only check for `reason=shipped_not_recorded`
(R6). The direct routes also get the §6.3 back-dating rule:
`validate_inventory_occurred_at(..., request)` runs `require_backdating` for a
named actor, so floor/office `POST /adjust {occurred_at: 15 d ago, backfill: true}`
is `403 BACKFILL_OWNER_ONLY` whatever the flag says. Every actor-reachable write
route is either in `ROUTE_ACTIONS` or named in `permissions.UNGATED_ROUTES`
(allocations, received-at, suppliers, intake extract/match, production runs,
notes, supply requests — no §4.3 row, open to every named role as today); a
test fails if a route is added to the allowlists without being placed.

**The two shared keys are not gated on the direct routes.** The master key
keeps every route it reaches today (A10 retires it); the dashboard key keeps
its route allowlist, which already includes order edits the matrix marks ✗ for
`legacy_dashboard` (cancel/close/reopen/status/header, ship commit) — gating
it would take those away from the live dashboard, which is "nothing new" the
wrong way round. Its ticket scope is still exactly the A1 part-1 grant.

**Note for the floor GPT:** `gpt-configs/schemas/openapi-floor.yaml` exposes
`POST /ship` (standalone) and `PATCH /sales/orders/{id}/status`, which §4.3
denies floor. A floor *actor key* now gets `ROLE_NOT_ALLOWED` on those two
operations; the master key is unaffected.

## Permission matrix (generated by `permissions.matrix_markdown()`, pinned by `tests/test_roles_a2.py`)

See the PR description or run `python -c "import permissions; print(permissions.matrix_markdown())"`.

## Migration 065 — staging first, prod only with owner approval

`migrations/065_entered_by.sql`: two nullable FK columns + one partial index +
marker `065_entered_by`. No backfill, no trigger change, no view replacement.
Rerunnable (`IF NOT EXISTS` / `ON CONFLICT DO NOTHING`). Down file:
`migrations/down/065_entered_by_down.sql`.

**Wrapper — both timeouts, one transaction, port 5432 (session pooler), as the table owner:**

```
BEGIN;
SET LOCAL lock_timeout = '5s';          -- how long to WAIT for each lock
SET LOCAL statement_timeout = '30s';    -- how long any one statement may RUN
SET LOCAL search_path = public;
\i migrations/065_entered_by.sql
COMMIT;
```

**What locks, and for how long.** `ALTER TABLE … ADD COLUMN` takes
ACCESS EXCLUSIVE on `transactions` and then on `ledger_corrections`, and the FK
takes SHARE ROW EXCLUSIVE on `actors`. Those locks are **held until COMMIT, not
until the statement ends** — so the index build and the FK validation that
follow run while `transactions` is still exclusively locked, and every reader
and writer of `transactions` (each `POST /make`, each dashboard page) queues
behind the whole transaction, not just the ALTER. `lock_timeout` only bounds the
wait to *acquire* each lock; it is `statement_timeout` that bounds the work done
while holding it. Keep both.

**Expected duration on production row counts** (read-only count 2026-10-08:
`transactions` 2,509 rows / 1.0 MB heap, `ledger_corrections` 6, `actors` 4;
PostgreSQL 17.6). Measured on a local scratch database seeded to those counts,
three runs: ADD COLUMN ≈ 0.7 ms each (catalog-only, no rewrite), index ≈ 0.2 ms,
FK ≈ 0.1 ms, marker ≈ 0.2 ms — **the whole transaction completes in ~2 ms**;
allow well under one second end to end with Supabase round-trips. The only way
it takes longer is waiting on a lock already held (a long dashboard read or an
open write): then `lock_timeout` fails the ALTER at 5 s, the transaction rolls
back cleanly (nothing applied, marker absent) and the wrapper is simply rerun.

Verify read-only: `entered_by_actor_id` on `transactions` and
`ledger_corrections`, FK to `actors`, index `transactions_entered_by_actor_idx`,
marker present, `trg_transactions_original_append_only` still `O`.

**Deploy order matters:** 065 must be applied **before** this code goes live on
an environment — every ledger INSERT names the column. Staging: apply 065, then
let Railway build the merge. Production: apply 065 by hand (same wrapper), then
merge. Rollback: revert the merge → wait for Railway → optionally the down file
(loses the actor-id attribution recorded since; `operator_id` keeps the name).

`tests/schema/schema.sql` ends with a pending `\ir ../../migrations/065_entered_by.sql`
block — remove after the prod apply + `scripts/dump_prod_schema.sh` (housekeeping
PR, as for 058/060/061).

## Tests

`tests/test_roles_a2.py` (264 cases): every (action × role) combination against
a hand-written copy of §4.3; the markdown table; prepare enforcement per key
for the five ticket actions; commit enforcement after a role change and after a
stored-action relabel; timing classification at 47 h / 48 h−1 m / 49 h /
14 d−1 h / 14 d+1 h and the 5-minute future grace; late entries for each
non-owner key (exception row, owner, detail, receipt); owner late entry without
exception; > 14 d denied for floor/office/dashboard with and without `backfill`;
owner/master > 14 d via `backfill:true`; future-dated rejection; commit-clock
reclassification (47 h → late, 13 d → denied); `entered_by_actor_id` on ticket
and direct writes (actor id vs `NULL` + `'legacy-shared-key'`), on void
corrections; whoami map = matrix; 065 up/down/up on an isolated database.
Review fixes: **direct routes** — office denied `POST /make|pack|adjust` with
the ticket's error shape and the gate answering before body validation; the
matrix per role on 12 more direct routes; close `shipped_not_recorded` owner-only;
owner/floor/master still open; shared keys unchanged; every actor-reachable
write route placed (gated or named exempt); floor > 14 d denied on `/adjust` and
`/make` with and without `backfill`; owner/master keep the backfill path; floor
49 h still posts. **Commit lock** — a concurrent `UPDATE actors SET role` blocks
(`lock_not_available` under a 500 ms `lock_timeout`) while a commit is between its
role check and its post, goes through after, and the next prepare by that key is
denied; an actor deactivated after prepare is `ACTOR_INACTIVE` at commit. **DST** —
`timing()` and `validate_inventory_occurred_at` across spring-forward (335.5 real
hours = late, where wall-clock math read 336.5 h = owner-only) and fall-back
(30 min in the future, which wall-clock math read as 30 min ago, is rejected),
on tickets (prepare and commit) and on the direct route; prepare and commit agree
on an entry prepared after fall-back and committed an hour later.
Existing tests updated for the rule: `test_write_tickets_part2.py` (office →
owner where office may no longer post; `test_part2_office_is_denied_by_role_not_route`),
`test_actor_attribution.py` (whoami shape; order-state attribution tests run as
office/owner, floor-schema scope test accepts the structured role denial),
`test_named_actor_writes.py` and `test_order_create_contract.py` (all-roles
parametrizations assert the denial where §4.3 applies; attribution assertions
run as an allowed role), `test_write_tickets.py` (race fixture applies 065).

## Coexistence with A5 (`origin/feat/lot-confirmation`, checked 2026-10-08 14:20)

A5 took migration numbers 062–064, so A2's migration is **065** (059 is already
a gap; numbers only need to be unique). A5 adds the `move_lot` ticket action
(`LOT` receipts, no ledger transaction): the matrix pre-registers `move_lot` for
owner / floor / office — **no shared key** (review fix 4: a new action is
"nothing new" for the master key too) — so A5's route is not denied for everyone
on merge, and the commit hook tolerates a response with no `transaction_id`.
A5's `tests/test_lot_confirmation.py::move` helper prepares/commits with the
master key (`headers()` default); after both land it must pass a floor or office
actor key, or those tests get `ROLE_NOT_ALLOWED`. Expected textual conflicts on merge, all trivial: the pending
`\ir` block at the end of `tests/schema/schema.sql`, the one-line migration
list in `isolated_database` (`tests/test_write_tickets.py`), `CHANGE_LOG.md`.
A5's `prepare()` test helper (auto-confirms lots) is imported by
`tests/test_roles_a2.py`, so the A2 tests keep working after A5 lands. A5's
`write_tickets.py` hunks sit between A2's (prepare `blockers`, commit
`effective_payload`, `record_identity`) with unchanged context lines on each side.

## Hook points in shared files (for the A5 lane)

`main.py`: `import permissions`; `_entered_by_actor_id()` next to
`_operator_id()`; `entered_by_actor_id` column + value on the 7 `transactions`
INSERTs and the 2 `ledger_corrections` INSERTs (plus an `entered_by_actor_id`
kwarg on `_append_transaction_correction` / `_append_transaction_line_correction`
and the two actor-reachable callers); `/auth/whoami` body. `write_tickets.py`:
`import permissions`; three marked `# A2 hook` blocks in `prepare()` and
`commit_ticket()`. `ticket_actions.py`: untouched.

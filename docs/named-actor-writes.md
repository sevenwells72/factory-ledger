# FR-15 named-actor writes

Owner decision, 2026-09-28: every active named actor can call all 14 routes
below, regardless of role. Permission enforcement remains in MCP. The shared
API key and dashboard key keep their existing behavior and scope. This
supersedes the dashboard-only actor scope in changelog row 142 for these routes.

The same actor-only policy also permits `POST /products/resolve`, the read-only
lookup used by MCP `office.resolveProducts`. The 2026-09-28 catalog audit found
it was the only MCP catalog route still blocked after the 14 write additions.
It uses the existing `verify_api_key` / `_authorize_api_key` mechanism and is
the only additional method/path pair authorized by this follow-up. The shared
key still succeeds, the dashboard key remains denied, and unrelated routes
(including administrative routes) retain their existing permissions.

## Root cause before the fix

Locations below refer to origin/main at `9daa732`, before this change.
All handlers call `verify_api_key` (`main.py:3114`), which delegates to
`_authorize_api_key` (`main.py:3080`). That check recognizes the named key and
loads its actor, but then requires membership in `DASHBOARD_KEY_ALLOWLIST`.
None of these 14 routes is in that list, so each gets HTTP 403 before writing.
The shared key returns success earlier in the same check and avoids that gate.

| Method and endpoint | Handler in main.py | Additional attribution gap |
|---|---|---|
| POST /receive | receive, line 5310 | INSERT omits operator; DB supplies legacy-shared-key |
| POST /ship | ship, line 7748 | INSERT omits operator; allocation helper receives True |
| POST /make | make, line 8082 | INSERT omits operator |
| POST /pack | pack, line 8589 | INSERT omits operator; allocation helper receives True |
| POST /adjust | adjust, line 8919 | INSERT omits operator |
| POST /void/{transaction_id} | void_transaction, line 9235 | Correction operator helper receives True |
| POST /customers | create_customer, line 11171 | No operator field or audit record |
| PATCH /customers/{customer_id} | update_customer, line 11195 | No operator field or audit record |
| PATCH /lots/{lot_code}/supplier-lot | update_supplier_lot, line 4935 | No operator field or audit record |
| PATCH /lots/{lot_id}/rename | rename_lot, line 5001 | No operator field or audit record |
| POST /sales/orders | create_sales_order, line 11403 | No creation operator field or audit record |
| POST /sales/orders/{order_id}/lines | add_order_lines, line 14611 | No creation operator field or audit record |
| PATCH /sales/orders/{order_id}/lines/{line_id}/cancel | cancel_order_line, line 14709 | Allocation releases already use actor-aware caller_source_tag, but a line with no allocation has no attribution |
| POST /sales/orders/{order_id}/ship | ship_order, line 14895 | Allocation releases already use caller_source_tag, but transaction INSERT omits operator |

`_operator_id` (`main.py:9046`) previously received the dependency's boolean
`True`, which it could not turn into an identity; it returned the placeholder
`legacy-shared-key`. Trace emitters also omitted their optional operator.

## Implementation and stored evidence

One change to the shared auth check adds `ACTOR_WRITE_ALLOWLIST` to the actor
branch only. It names exactly these 14 write route templates plus the
`POST /products/resolve` lookup. Existing dashboard routes remain available
to actors; unrelated admin routes remain excluded.
There are no backend role checks, no new credentials, and no historical backfill.

The handlers use the actor already resolved on `request.state`. Inventory
transactions set `transactions.operator_id` at INSERT, including each physical
line shipped against an order. Void sets `ledger_corrections.operator_id`,
leaving the original immutable transaction untouched. Trace events use the same
name, including void markers; ship/pack allocation releases also carry it.
Request-body identities cannot override the authenticated actor.

Customer, lot and order metadata edits do not create inventory transactions.
Migration 056 adds `actor_write_audit` to record the stable actor ID, a snapshot
of the actor name in `operator_id`, method, route template, target table/ID and
database timestamp. Order creation records both the order and its lines.
The audit insert shares the business transaction, fails closed, and is protected
from UPDATE/DELETE by the existing append-only trigger. No secrets are stored.
No-op lot renames produce no audit because they make no business change.

Shared keys still return True before actor lookup; their transaction and
correction operators remain `legacy-shared-key`, trace operators remain NULL,
and existing allocation source tags remain unchanged. Metadata audit calls
issue no SQL for legacy keys, so shared-key writes still work without 056.
Response shapes, auth status codes, key caching, and revocation are unchanged.

## Validation and rollout

Tests exercise valid HTTP requests and read back actual PostgreSQL rows for all
14 routes with Blubber (owner), Arturo and Luz (floor), Miriam (office), and the
shared key. They also cover missing, invalid, inactive and dashboard keys on
every route; spoofed body identities; preview purity; atomic rollback when an
audit or trace write fails; alternating actor/shared calls; allocation releases;
legacy writes without the new table; and migration rerun/append-only behavior.
Product-resolution coverage checks all four named actors and the shared key,
the unchanged lookup response and business data, rejection of missing, invalid,
inactive and dashboard keys, and the exact actor-only allowlist scope.
Everything runs against local TEST_DATABASE_URL with savepoint rollback.
Validation: **1,323 Python tests passed** (baseline 1,153; 170 new), **67 Node
tests passed**, zero failures/skips, and `git diff --check` clean.

Migration 056 must be applied after 052 and before a later authorized rollout
of named-actor metadata writes. It has only been run inside local test
transactions. Production has not been accessed or changed. Deploy and merge
are outside this task.

Future consideration only: if authorization moves into the backend, review
void, adjust, and lot rename first because they can alter inventory availability
or traceability. No such role restrictions are added here.

## PR #66 review fixes (2026-09-29)

- **Shortcut routes.** The ten `/{receive,ship,make,pack,adjust}/{preview,commit}`
  wrappers now pass the request to their handler. Before, an actor-keyed
  `POST /receive/commit` (reachable through the dashboard allowlist) stored
  `legacy-shared-key` and a NULL trace operator. HTTP scope is unchanged: the
  ship/make/pack/adjust shortcuts still return 403 to actor keys.
- **Auto-created customers.** `resolve_customer_id(..., request=)` writes a
  `customers` audit row on the same cursor when it creates a customer, so the
  customer that `POST /sales/orders` or `POST /ship` makes as a side effect is
  attributed, and a failed audit rolls back the order or shipment too.
- **Row-level security.** Migration 056 enables (does not force) RLS with no
  policies. The backend role must own the table, so apply 056 as the same
  role the app connects as. Other non-superuser roles see and write nothing.
- **Rollback.** `migrations/down/056_actor_write_audit_down.sql` drops the table
  and marker; see "Rollback" below for the guard and the exact procedure.

Validation after these fixes: **1,347 Python tests passed** (24 new), **67 Node
tests passed**, zero failures/skips, on a throwaway local database. The 14
F1/F2 regression tests fail against the pre-fix code.

## Production rollout (2026-09-29 / 2026-09-30)

Executed with owner approval for each production step; all times UTC.

- **056 applied 2026-09-29 16:53:55** as `postgres` (table owner) over the
  session pooler on port 5432, inside `BEGIN; SET LOCAL lock_timeout='5s'; ...
  COMMIT;` with `ON_ERROR_STOP`. Verified read-only: `actor_write_audit` owned
  by `postgres`, RLS enabled and not forced, no policies, 0 rows, append-only
  trigger enabled, both indexes valid, marker present, `ledger_current_*` view
  definitions unchanged.
- **057 applied 2026-09-29 16:56:23** the same way (see
  `docs/order-create-contract.md`). The old backend kept serving identical order
  and packing-slip responses with both migrations in place.
- **PR #66 merged 2026-09-29 16:58:36** as merge commit `67db9a0`. Railway
  deployment `353e5aed` sat QUEUED for ~30 minutes during a Railway incident
  ("API degradation causing slow or stuck deployments") and went live ~17:29;
  the new deployment's log showed the verification requests and the previous
  deployment was removed.
- **PR #67 merged 2026-09-29 17:30:38** as `521330c`. Railway never received
  that push (same incident); Netlify deployed the dashboard (`dashboard.js?v=68`)
  at ~17:33, which is backward-compatible with the #66 backend. The backend was
  deployed 2026-09-30 13:53 via `railway redeploy --from-source` (deployment
  `c4f4bca5`, commit `521330c`) and verified: `/health` 200, legacy NULL-price
  lines still return null price/value, packing slip unchanged apart from its
  generated timestamp, `sales_order_create_receipts` 0 rows, dashboard loads.
- **Not exercised during rollout:** no named-actor request was sent (no actor
  key was at hand) and no test writes were made; `actor_write_audit` was 0 rows
  at every check. The first real named-actor write is the owner's.
- Housekeeping (this branch): `tests/schema/schema.sql` re-dumped from
  production after both migrations and the temporary `\ir` includes for 056/057
  removed.

## Rollback

Order: revert the PR #67 merge (`git revert -m 1 521330c`), then the PR #66
merge (`git revert -m 1 67db9a0`), wait for Railway to deploy the reverted
backend, then run `migrations/down/057_order_create_contract_down.sql`, then
`migrations/down/056_actor_write_audit_down.sql`. Both down scripts run as the
table owner over port 5432 with `ON_ERROR_STOP`; 057 needs an explicit
`BEGIN; SET LOCAL lock_timeout='5s'; \i ...; COMMIT;` wrapper, while the 056
script carries its own `BEGIN`/`COMMIT` and must not be nested in another
transaction.

The 056 script guards the attribution history:

- It takes `LOCK TABLE public.actor_write_audit IN ACCESS EXCLUSIVE MODE` so
  no named-actor write can slip in between the emptiness check and the drop.
- It **refuses to run while the table has rows**, raising
  `056 rollback refused: actor_write_audit is not empty ...`, because dropping
  the table permanently deletes who edited which customer, lot, order and line.
- After exporting the rows and verifying the export, the operator overrides the
  refusal by adding `SET LOCAL factory_ledger.confirm_audit_export = 'yes';`
  inside the script's transaction (after `BEGIN`, before the `DO` block). The
  setting clears at `COMMIT`/`ROLLBACK`; no other value permits the drop.
- Revert the backend first: while code that calls `_record_actor_write()` is
  deployed, every named-actor metadata write fails closed (500 + rollback) once
  the table is gone. Shared-key writes never touch the table.

Post-rollout validation on this branch, against a throwaway local database
built from the re-dumped schema: **1,443 Python tests passed**, **69 Node
tests passed**, zero failures/skips.

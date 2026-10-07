# A1 part 1 — receive tickets and receipts

Status: implemented and tested; migration 058 applied to staging only on
2026-10-07. No hosted application deployment, production access, merge, or key
rotation. Branch `feat/write-tickets` was created from `cb2705c` and rebased onto
`origin/main` at `d6ead6c` after documentation-only PRs #74/#75 landed. The A1
code and migration match the tested/staging version byte for byte. F1 backend
ticket custody and A11 PIN sessions remain separate chunks; the plain A1
endpoint contract is unchanged.

## Review boundary

This is the first reviewable part of A1. It delivers migration 058, the shared
receive preview/post core, `POST /receive/prepare`, atomic
`POST /tickets/{ticket}/commit`, three receipt read routes, and an explicit
expiry script. Existing direct route behavior and permissions remain available.

Remaining A1 part 2: make, pack, adjust and found ID adapters and prepare routes;
their same-cursor commit integrations, lot/input snapshots and duplicate-match
rules; action-specific and insufficient-stock revalidation tests; staging smoke
receipts for those four actions. The five-action schema and receipt prefixes
are ready, but only receive can currently issue or commit a ticket.

Roles, correction reasons, lot confirmations, shipping/BOL and order tickets,
resolution changes, dashboard screens and client adapters remain later chunks.

## Contract

- Receive input uses `product_id`, optional `supplier_id`, `cases`,
  `case_size_lb`, `bol_reference`, and the existing receive lot/backfill fields.
  Commingled entries use `supplier_id` too. Names return `422 IDS_REQUIRED`.
  Unknown extra fields are rejected so the draft cannot promise unposted data.
- `occurred_at` is frozen at prepare; `happened_at` is an accepted alias.
  The displayed lot code and expected-receipt match are pinned in the stored
  payload. A closed or reassigned expected receipt requires a fresh prepare.
- `client_source` is `api` by default. `api`, `mcp`, and `fl_assistant` expire
  after 10 minutes; dashboard forms expire after 30 minutes. No client-specific
  ticket fields or business-rule branches exist.
- Prepare returns a ticket even for business-validation blockers. Warning
  `POSSIBLE_DUPLICATE` requires acknowledgement at commit; it checks effective
  posted receives across all actors over the preceding 24 hours.
- Commit accepts only `payload_hash`, `acknowledged_warnings`, and optional
  `client_source`. The stored prepare source remains the audit source.
  Receipt/counter/lot/ledger/trace changes share one transaction. Rejected and
  expired statuses persist; unexpected failures roll back to prepared.
- Replays retain the saved response and receipt, changing only `replayed` to
  true. Caller and payload binding are checked before replay.
- Receipt reads: `GET /receipts/{receipt_number}`, `GET /receipts`, and
  `GET /receipts/by-transaction/{transaction_id}`. Unknown or pre-ticket
  transactions return JSON `404 RECEIPT_NOT_FOUND`.
- List filters are `date`, `actor` (exact name or ID), `action`, `status`
  (ticket status), and `client_source`. A receipt appears on both its happened
  plant day and its entered plant day, with `late_entry` when these differ.
  Receipt details expose the ledger's current effective status and lines.

Both auth allowlists gain exactly the five routes above. No existing entry is
removed. No actor role enforcement is introduced.

## Decisions and design differences

The user's build overrides take precedence over design §1.7/§11 decision 1:
direct writes remain available until A10. PR-0 rotation is deferred to cutover
day, not a build precondition. Design docs are left to the separate docs worktree.

The prepared A1 prompt's compatibility contract permits nullable `actor_id`
for legacy keys; `operator_id` and an added `key_kind` column bind those tickets
to the preparing legacy identity. A CHECK enforces actor/key-kind consistency.
Role/name are captured in the draft for receipt history.

`write_tickets.receipt_number` is unique; `transactions.receipt_number` has a
non-unique index, following §2's future multi-transaction shipment receipt.
Committed tickets reject UPDATE and DELETE. The down migration refuses populated
ticket/counter tables unless an exported-and-verified evidence override is set.
The application role owns both RLS-enabled tables; no public policies are added.

The receive core is extracted without a connection override or nested route
call. Ticket validation uses it and the existing lot-code identity guard.
Only `active=false` archives a product, preserving existing nullable-active
catalog behavior. No original ledger view or historical migration is changed.

## Verification

Fresh local PostgreSQL 17 database, rebuilt with `scripts/setup_test_db.sh --fresh`.
The existing Python 3.12 test environment was reused through a temporary
`.venv-test` symlink, removed after verification. For a new environment, use
Python 3.12 and install both `requirements.txt` and `tests/requirements-test.txt`.
Tests ran with `TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_test`
explicitly set and inherited `DATABASE_URL` removed before pytest:

- Full Python suite: **1,526 collected, 1,526 passed**, zero failures/skips.
- Database-marked tests: **942 collected, 942 passed** in that full run.
- New ticket tests: **54 passed**, including real concurrent HTTP commits on a
  dedicated temporary local database, replay, expiry, payload/user binding,
  rollback after ledger insertion, revalidation, duplicate warnings, counter
  boundaries, receipt reads, RLS, immutability and migration down/up/rerun.
- `test_staging_safety.py` and `test_seed_staging.py` run unchanged in the suite.
- Node suite (`node --test tests/*.js`): **69 passed**, zero failures/skips.
- A full rerun needs a fresh DB: existing seed tests assume an otherwise empty
  reserved fixture-ID range, and other existing tests advance sequences.

Staging-only local-server smoke, 2026-10-07 15:34 ET:

| Receipt | Transaction | Ticket | Lot | Quantity |
|---|---:|---:|---:|---:|
| RCV-261007-001 | 1000000003 | 1 | 1000000004 | 0.01 lb |

Fixture reference `STG-A1-7DA20A23D052`; synthetic actor `1000000004`.
Prepare, commit, identical replay, receipt lookup and reverse lookup passed.
The local server used a free port because 8765 was occupied; that process was
left alone. All disposable actors are inactive and their sequence is preserved.
No hosted staging service configuration was changed. Credentials were loaded
from the protected staging files and never printed or stored in this checkout.

## Operations and owner steps

Migration 058 is idempotent, with marker `058_write_tickets`. Startup uses the
existing serialized marker gate; the standalone SQL runner also writes the
marker. No expiry or new data sweep runs at startup.

For an explicit schema application, use the app/table owner, port 5432, an
explicit transaction, `ON_ERROR_STOP`, `SET LOCAL lock_timeout='5s'`, and
`SET LOCAL search_path=public`, then include `migrations/058_write_tickets.sql`.
Revert the application before any optional down migration. The down file is
`migrations/down/058_write_tickets_down.sql`; read its evidence-export guard.

Manual local expiry (idempotent; no job is scheduled in this PR):

```sh
DATABASE_URL=postgresql://localhost:5432/factory_ledger_test \
  .venv-test/bin/python scripts/expire_tickets.py
```

For staging, load the protected URL into the process environment and set the
same staging identity guards as the application. The script currently accepts
only localhost or explicitly guarded staging. Production scheduling/operation
is outside this part.

Owner steps for a later authorized rollout: review this part and the remaining
A1 work; apply 058 to production before deploying the code; re-dump
`tests/schema/schema.sql` and remove the entire pending 058 block; add a
`FACTORY_LEDGER_CHANGELOG.md` deployment row at actual deployment time. Rotation
remains an owner cutover-day step. This PR must not be merged by the agent.

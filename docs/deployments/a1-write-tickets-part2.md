# A1 part 2 — make, pack, adjust and found tickets

Built from merged part 1 (`442c675`) in the new `feat/write-tickets-2`
worktree, then rebased onto `f3f5cf8` after A4 resolution and schema housekeeping
merged concurrently. All upstream migrations/schema changes are preserved; the
part-2 diff adds none. The previous A1 worktree was clean: its local-only commits contained
part-1 implementation/documentation, with no part-2 changes to salvage.
No reset or cherry-pick was needed. No migration or schema include is added or
changed; this uses the existing migration 058 tables and receipt counters.

## Contract

Four actor/master-authenticated POST prepare routes use the existing ticket envelope,
10-minute expiry (30 minutes for `client_source=dashboard`), identity/hash binding,
supersession, warning acknowledgement, atomic commit and saved-response replay:

| Route | Required input | Optional action input |
|---|---|---|
| `/make/prepare` | `product_id`, positive integer `batches` | `lot_code`, `ingredient_lots: [{ingredient_product_id, lot_id}]`, `excluded_ingredients: [product_id]`, `confirmed_sku` |
| `/pack/prepare` | `source_product_id`, `target_product_id`, positive integer `cases` | `case_weight_lb`, `lot_allocations: [{lot_id, quantity_lb}]`, `target_lot_code` |
| `/adjust/prepare` | `lot_id`, nonzero signed `delta_lb`, `reason_code` | `reason_es`; `reason` is accepted as an input alias |
| `/inventory/found/prepare` | `product_id`, positive `quantity`, `reason_code` | `uom: "lb"`, `lot_code`, `found_location`, `estimated_age`, `suspected_supplier`, `notes`, `notes_es` |

Adjust persists only `reason_code` in the ticket payload and draft, aligned with
PR #85's `transactions.reason_code` vocabulary; the legacy posting adapter maps
it to `AdjustRequest.reason`. No transaction-column migration or reason catalog
validation is added here; A3 owns that validation. Supplying both aliases is rejected.

The public dashboard key retains exactly the part-1 ticket scope: receive
prepare, ticket commit and receipt reads. It cannot prepare make, pack, adjust
or found tickets. Named actor keys and the master key can use all four routes;
commit identity binding remains enforced.

All accept `occurred_at` (or `happened_at`), `backfill`, and `client_source`.
Unknown fields, caller-supplied identities, name-based product selection and
client-supplied internal plans are rejected. Found supports pounds explicitly;
it does not silently convert another unit into pounds.

Commit is still `POST /tickets/{ticket}/commit` with only `payload_hash` and
optional `acknowledged_warnings`. The receipt reads are unchanged. Receipts
use `MK`, `PK`, `ADJ`, or `FND`, and list both input and output lot IDs.
Named actors are attributed on ledger and trace writes; found audit attribution
comes from the authenticated actor. Direct routes retain their prior request,
response, permission and posting behavior.

## Revalidation and races

Make and pack drafts record the exact input lot IDs and quantities, including
pack add-ins and make plans spanning several FIFO lots. Revalidation locks
those lots and checks current effective posted stock, active products, merged
lot status, output quantities and recipe requirements. A balance change that
still supports the draft succeeds with `state_changed: true`. A shortage,
changed recipe, archived product or merged input rejects the ticket without
posting or consuming a receipt number. Adjust keeps its existing signed-count
semantics, including negative-balance warnings; the later shortage/reason
policy is not introduced here.

Make/found pin generated codes; pack pins its inherited source code. If a draft
showed a new output lot but a competing legacy or ticket write takes that code,
commit returns `409 TICKET_STALE` with blocker `LOT_CODE_TAKEN`. A second check
at `find_or_create_lot` closes the gap after validation, including creators that
do not share the action's sequence lock. A fresh make/found prepare generates
a new code. Pack inherits the same source code, so its fresh draft explicitly
shows that output lot as existing. Existing output lots are pinned by ID too:
a rename/replacement requires a fresh draft. Adjust never generates a code;
its ID binding survives a rename even if another lot takes the old code.

Posting cores use the caller's cursor. Receipt allocation, ledger lines,
ingredient consumption, trace events and the saved ticket response all commit
together. Same-ticket concurrent calls serialize and return one saved receipt.
Unexpected errors roll back to prepared; validation rejection persists.

Duplicate warnings use current effective posted ledger entries across all
actors: make matches product and output pounds within 45 minutes; pack matches
target and cases within 45 minutes; adjust matches lot and signed pounds within
2 hours; found matches product and pounds within 2 hours. Amounts match ledger
numeric precision. Ticket packs use stored cases; legacy packs use the anchored
`Pack N cases of ...` note written by the existing core because there is no
legacy pack cases column. Warnings name the product and the earlier transaction’s lots, and carry its
receipt (or legacy transaction), actor and elapsed minutes, and require acknowledgement.

A2 role enforcement, A3 reason catalog/shortage rules, A4 resolution,
A5 lot confirmations/supplier tracking, A10 direct-route cutover, A11 PIN sessions and A12 kosher
attestation remain separate work. No GPT schema, dashboard or role policy is
changed here.

## Nightly ticket expiry

`scripts/expire_tickets.py` already implements the shared, idempotent sweep:

```sql
UPDATE write_tickets SET status = 'expired'
WHERE status = 'prepared' AND expires_at < now();
```

Railway cron service setup (documentation only; no service is created by this PR):

1. After the reviewed code is deployed, add a separate cron service using the
   same repository and deployed revision as FastAPI, in the intended Railway
   environment. Keep the FastAPI web service's start command unchanged.
2. Set the cron schedule to **`15 7 * * *`** (daily at 07:15 UTC, 03:15 EDT /
   02:15 EST). Set its start command to **`python scripts/expire_tickets.py`**,
   with the repository root as the working directory. Include `/scripts/**`
   in this cron service's watch paths (the shared `railway.json` only watches
   root Python files). The process exits after one sweep; no loop or web listener
   is needed. See [Railway cron jobs](https://docs.railway.com/cron-jobs).
3. In the cron service's Variables settings, reference `DATABASE_URL` from
   the matching FastAPI service rather than copying its value into Git or the
   start command. For a service named `FastAPI`, use Railway's variable
   reference `${{FastAPI.DATABASE_URL}}`; select the actual service name in
   Railway if it differs. Use FastAPI-staging for the staging cron. See
   [Railway reference variables](https://docs.railway.com/variables#referencing-another-services-variable).
4. For production, set **`ENVIRONMENT=production`** and
   **`PRODUCTION_DATABASE_HOST`** to the database hostname used by that
   FastAPI service. The parsed connection host must match this configured
   hostname; an absent environment/host or a mismatch refuses the connection.
   For staging, set `ENVIRONMENT=staging`, `PRODUCTION_DATABASE_HOST`,
   `STAGING_DATABASE_HOST` and `STAGING_DATABASE_PROJECT_REF` to match
   FastAPI-staging's existing isolation guard.
5. Do not set libpq routing overrides (`PGHOSTADDR`, `PGSERVICE`,
   `PGSERVICEFILE`) or URL routing parameters such as `host`, `hostaddr`,
   `service`, `user` or `dbname`. The guards reject these before connecting.
   Never put a URL/key in command output. Monitor failed runs and rerun after
   recovery; the sweep is idempotent.

The script prints only an expired-row count or exception class and returns
nonzero on failure. Committed, rejected and superseded evidence is never
changed or deleted. Commit also enforces expiry synchronously, so scheduler
downtime cannot make an expired ticket usable. No startup migration marker
gates the sweep. This change does not create a hosted job or access production.

## Verification

- Final PR #83 review fixes: **1,791 Python tests passed**, zero failures/skips,
  on a fresh PostgreSQL 17 database (`fl83_final` at loopback port 57683).
  This includes 62 added regression cases. The earlier rebased suite passed
  1,729 tests, and the pre-rebase suite passed 1,606.
- Part-2 safety tests: **94 passed**; expiry guard tests: **39 passed**.
  Focused ticket/auth/expiry coverage passed 206 tests before the final
  prior-transaction lot-label correction; the final full run covers that correction.
- A repeated full run against the already-used test database hit the existing
  seed test's fresh-database assumption (15 rows instead of 3). The final run
  used a newly created database and passed every test. No test was skipped.
- The dashboard allowlist equals the part-1 `origin/main` set exactly (89 total
  routes, including its five ticket routes). All three deleted assertions are
  restored. Named actors/master retain the four additional prepare routes.
- Adjust payloads and draft receipts expose `reason_code`; both input aliases
  produce the same payload hash. Replay and arbitrary reasons remain supported.
- Blockers/warnings identify product names and lot codes, including prior
  transaction lots in duplicate warnings. Example: “Oats 50 lb needs 10.0 lb;
  prepare again after resolving the shortage.”
- Node suite: **69 passed**, zero failures/skips.
- New PostgreSQL tests cover every action's single use, expiry, tamper,
  revalidation, replay, actual concurrent double commit and duplicate warnings;
  also legacy/ticket code races, the post-validation creation race, concurrent
  different-ticket code races, strict inputs, recipe changes, exact multi-lot
  consumption, pack add-ins, receipt reads, actor binding and trace rollback.
- Existing receive, direct-route, staging isolation and startup suites remain
  included. The existing 058 migration and test schema are byte-for-byte
  unchanged. Tests explicitly remove inherited `DATABASE_URL` and set a
  dedicated local `TEST_DATABASE_URL`; bytecode/pytest cache writes are disabled.

Staging smoke on 2026-10-08 used a temporary loopback HTTP server running this
branch against the protected staging database, with startup migrations/sweeps
disabled. The hosted staging service/configuration was not changed. Reference
`STG-A1P2-57FB94002B5A`; synthetic actor `1000000006`, deactivated afterward.
All fixtures are synthetic and retained as evidence. No production access.

| Action | Receipt | Transaction | Ticket | Posted quantity |
|---|---|---:|---:|---:|
| Make | MK-261008-002 | 1000000010 | 6 | 0.10 lb output |
| Pack | PK-261008-002 | 1000000011 | 7 | 0.05 lb output |
| Adjust | ADJ-261008-002 | 1000000012 | 8 | -0.01 lb |
| Found | FND-261008-002 | 1000000013 | 9 | 0.01 lb |

Each prepare, commit, identical replay, receipt lookup and reverse lookup
passed; direct staging SELECTs confirmed exactly one attributed transaction per
ticket. This smoke ran again after the final ID-binding/decimal-match review
changes; the earlier smoke receipts ending in `001` are also retained. No hosted deployment, merge,
production access, key rotation or production migration was performed.

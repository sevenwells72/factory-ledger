# A1 part 2 — make, pack, adjust and found tickets

Built from merged part 1 (`442c675`) in the new `feat/write-tickets-2`
worktree, then rebased onto `f3f5cf8` after A4 resolution and schema housekeeping
merged concurrently. All upstream migrations/schema changes are preserved; the
part-2 diff adds none. The previous A1 worktree was clean: its local-only commits contained
part-1 implementation/documentation, with no part-2 changes to salvage.
No reset or cherry-pick was needed. No migration or schema include is added or
changed; this uses the existing migration 058 tables and receipt counters.

## Contract

Four authenticated POST prepare routes use the existing ticket envelope,
10-minute expiry (30 minutes for dashboard), identity/hash binding,
supersession, warning acknowledgement, atomic commit and saved-response replay:

| Route | Required input | Optional action input |
|---|---|---|
| `/make/prepare` | `product_id`, positive integer `batches` | `lot_code`, `ingredient_lots: [{ingredient_product_id, lot_id}]`, `excluded_ingredients: [product_id]`, `confirmed_sku` |
| `/pack/prepare` | `source_product_id`, `target_product_id`, positive integer `cases` | `case_weight_lb`, `lot_allocations: [{lot_id, quantity_lb}]`, `target_lot_code` |
| `/adjust/prepare` | `lot_id`, nonzero signed `delta_lb`, `reason` | `reason_es`; `reason_code` is accepted instead of `reason` |
| `/inventory/found/prepare` | `product_id`, positive `quantity`, `reason_code` | `uom: "lb"`, `lot_code`, `found_location`, `estimated_age`, `suspected_supplier`, `notes`, `notes_es` |

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
legacy pack cases column. Warnings carry the earlier receipt (or legacy
transaction), actor and elapsed minutes, and require acknowledgement.

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

Configure a staging scheduler to run **once nightly, for example 07:15 UTC**,
with this repository as its working directory and this command:

```sh
python scripts/expire_tickets.py
```

Supply the protected staging `DATABASE_URL` through the scheduler's secret
environment, plus `ENVIRONMENT=staging`, `PRODUCTION_DATABASE_HOST`,
`STAGING_DATABASE_HOST`, and `STAGING_DATABASE_PROJECT_REF`, matching the
existing staging service. Never put the URL/key in the command or job output.
The script refuses nonlocal databases without the staging guard, returns
nonzero on failure, and prints only an expired-row count or exception class.
Alert on a nonzero exit and rerun safely after recovery. Committed, rejected
and superseded evidence is never deleted or changed. Expiry is also enforced
synchronously at commit, so scheduler downtime cannot make an expired ticket
usable. No startup migration marker gates this sweep. This PR documents the
schedule; it does not install a hosted job or change production scheduling.

## Verification

- Full Python suite: **1,606 passed**, zero failures/skips, on a dedicated fresh
  PostgreSQL 17 database (`fl_a1p2_final` at loopback port 57682).
- Part-2 safety tests: **72 passed**, including the final review regressions.
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

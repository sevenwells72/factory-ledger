# A3b — Exceptions enforcement (reason codes, large-correction holds, shortage post-and-flag, `/exceptions`)

Builder: Claude Code. Reviewer: Codex. **Do not merge.**
Base: `origin/main` 2e9432b (PR #92). Branch `feat/exceptions-enforcement`, worktree `~/dev/fl-a3b-wt`.
Design: `docs/design/phase1-safe-operating-system.md` §5 (R2, R3), §5.1, §7.1, §11 item 7; FOLLOWUPS P1.8.
Owner decisions (Michael, 2026-10-08) are quoted in the module docstring of `exceptions_enforcement.py`.

## What it does, in plain English

* **Every correction needs a reason** from the fixed list of 8 (`correction_reasons`, 061). The
  ticket refuses a bad reason outright (`422 REASON_CODE_INVALID`, with the valid list); `unknown`
  needs a note (`422 NOTE_REQUIRED`); `damage_disposal`/`unrecorded_usage` only remove stock and
  `missing_receipt` only adds it (`422 REASON_SIGN_MISMATCH`). Legacy codes a client still sends
  (`damage`, `count_correction`, `found_back_stock`, …) are translated through the 061 map and the
  translation is shown in the draft. `transactions.reason_code` is written at INSERT for tickets
  **and** for the direct legacy routes (mapped the way the 061 backfill mapped history), so P1.8's
  NULL window closes; 069 sweeps the rows inserted during the window once.
* **Big corrections are highlighted**: |Δ| > 500 lb, or > 10 % of the lot's **book balance just
  before** the correction (read under lock at commit), or any correction on a lot whose book
  balance is ≤ 0. Found stock uses the 500 lb rule only and is always on the weekly view. The
  facts travel on the receipt as `correction_review` (`rules`, `book_balance_before`,
  `pct_of_book`, `lb_equivalent`, `highlighted`, `photo_required`, `attachment_ref`).
* **> 500 lb needs a photo.** With an `attachment_ref` (prepare body or commit body) it posts,
  highlighted. Without one, commit returns **202** `{held: true, status: "awaiting_approval",
  exception_id}`: nothing posts, the ticket is held (it never expires), and one
  `exceptions(LARGE_CORRECTION, block)` row is opened for the owner, linked to `ticket_id` and
  the `payload_hash`. Repeating the commit without a photo returns the same hold.
* **Approval is the commit.** `POST /exceptions/{id}/approve` (owner only) runs one
  transaction: lock exception + ticket, verify held + hash, re-validate current state, post
  through the same commit path with **`entered_by` = the preparer**, record
  `approved_by` = owner on the response and the exception, mark both done. A replay returns the
  original receipt; two concurrent approvals post once. A stale draft (lot merged, product
  inactive, preparer deactivated) → `409 TICKET_STALE`, the hold and exception stay open for the
  owner to reject explicitly. `POST /exceptions/{id}/reject` (owner, note required) → ticket
  `rejected`, exception `resolved/declined`, nothing posts. A later photo on the normal commit
  path also releases the hold — once.
* **Insufficient stock never blocks a make or pack.** It is the only non-blocking condition:
  prepare warns `WILL_CREATE_SHORTAGE` (no acknowledgement needed), commit recomputes the
  shortfall under `FOR UPDATE`, posts the full consumption against the confirmed/pinned lot (its
  balance goes negative), and in the same transaction writes one `shortage_flags` row and one
  `exceptions(SHORTAGE, warn)` per short lot, owner = floor, due in **2 business days**. Replays
  return the stored receipt; the 069 unique indexes make a second flag impossible. Everything
  else still blocks exactly as before (A5 lot confirmation, A2 permissions and back-dating, lot
  merged, lot code taken, warnings not acknowledged). A product with **no lot at all** still
  blocks (`INSUFFICIENT_STOCK`): there is nothing to go negative.
* **Never add stock to cover a shortage**: a positive adjust on a lot with an open shortage, or a
  found for that product, is `409 SHORTAGE_OPEN_RESOLVE_INSTEAD` naming the exception. Resolve
  it first (`counted` / `missing_movement` / `voided`, note required, who/when recorded), then
  correct the stock.
* **Deadlines** count Mon–Fri after the entry day, due 23:59 America/New_York: shortage 2, unidentified
  lot 7. One helper (`business_deadline`); A5's `identification_deadline` now delegates to it.
  Plant-closure days: FOLLOWUPS P1.12.
* **`/exceptions` routes** (named actors only; master and dashboard keys get `403`):
  `GET /exceptions?kind=&status=open|escalated|resolved|waived|all&overdue=&owner_actor_id=&product_id=&lot_id=`
  (default = the open queue incl. escalated, overdue first), `GET /exceptions/{id}`,
  `POST …/resolve {resolution_kind, note, receipt_number?}` (owner+floor; `LATE_ENTRY` /
  `SHIPMENT_PROOF_MISSING` owner only; kind-specific `resolution_kind` lists), `POST …/approve
  {note?}` and `POST …/reject {resolution_kind:"declined", note}` (owner). Every write is
  attributed in `actor_write_audit` (069 widens its `target_table` CHECK).
* **Nightly sweep** `scripts/exceptions_sweep.py`: escalates overdue open exceptions and shortage
  flags (`status → escalated`, `escalated_at`), idempotent, same database guard as
  `expire_tickets.py`. Railway cron documented below, **not created**.

## Files

| File | Change |
|---|---|
| `exceptions_enforcement.py` | **new** — all A3b logic (reasons, review, shortfall, flags, hold/approve, routes, deadline helper) |
| `migrations/069_exceptions_enforcement.sql` (+ `down/`) | **new** — see below |
| `ticket_actions.py` | hooks only: `choose_inputs` shortfall pin; adjust/found reason + cover-up + review; pack shortfall; commit-time `shortfalls()`; `post()` records shortages and the review |
| `write_tickets.py` | `note`/`attachment_ref` fields; `WILL_CREATE_SHORTAGE` warning; held tickets committable/never expire; the posting block extracted **verbatim** into module-level `execute_commit()` (shared with approval); `PREPARE_REFUSALS` raise instead of drafting |
| `main.py` | `import exceptions_enforcement`; `ACTOR_WRITE_ALLOWLIST … \| exceptions_enforcement.ACTOR_ROUTES`; `_post_prepared_inputs` no longer calls `validate_lot_deduction` (ticket path only); `_pack_commit_core` ticket path accepts pinned ≤ 0 lots and skips the sufficiency 400s; `_adjust_commit_core(…, reason_code, note)` / `_found_commit_core(…, reason_code)` write `reason_code` at INSERT (direct routes: 061 mapping); found-with-new-product INSERT likewise; `register_routes` at the end |
| `permissions.py` | `ROUTE_ACTIONS` + 3 `/exceptions` writes; EN/ES labels for the three A2 exception actions. **No new actions, no matrix change** |
| `lot_confirmation.py` | `identification_deadline()` delegates to `business_deadline(…, 7)` |
| `scripts/exceptions_sweep.py`, `scripts/check_exceptions_staging.py` | new |
| `tests/conftest.py`, `scripts/setup_test_db.sh` | seed the fixed reason list (061 §1–2) into the schema-only test DB, committed once, idempotent |
| `tests/schema/schema.sql` | pending `\ir …/069_exceptions_enforcement.sql` tail (remove after the prod apply + re-dump) |
| `tests/test_exceptions_enforcement.py` | new (see Tests) |
| existing tests | `test_write_tickets_part2` (adjust fixture uses a legacy code; shortage tests now expect post+flag; add-in shortage flags), `test_write_tickets` (isolated DB seeds reasons + 069; 058 up/down rolls 069 back first), `test_roles_a2` (065 up/down likewise), `test_named_actor_writes` (actor scope lists the 5 routes) |
| `FOLLOWUPS.md` | P1.8 closed; P1.12 plant-closure days; P1.13 attachment storage dependency; P1.14 sweep scope + cron |

## Migration 069 (`migrations/069_exceptions_enforcement.sql`, down file included)

Additive, rerunnable, **staging first**, production only with owner approval and **before** the
code goes live there (the code writes `write_tickets.status='awaiting_approval'`, relies on the
unique indexes, writes `actor_write_audit.target_table='exceptions'`, reads `reason_code` through
the view). 069 was the next free number: 066 (supplier labels, PR #91 on hold), 067 (A7, PR #93),
068 (F1, PR #90) are all taken.

1. `write_tickets.status` CHECK + `'awaiting_approval'` (rebuild, full known list) + partial index.
2. Unique indexes: `shortage_flags(transaction_id, lot_id)`; `exceptions(transaction_id, lot_id) WHERE kind='SHORTAGE'`; `exceptions(ticket_id) WHERE kind='LARGE_CORRECTION'`.
3. `CREATE OR REPLACE VIEW ledger_current_transactions` — identical body, **two trailing columns** `reason_code`, `entered_by_actor_id` (P1.8 / P1.11a). Dependent views untouched.
4. One-time sweep of `transactions.reason_code IS NULL` adjust rows (the P1.8 window), 061's four tiers, 061's single-statement trigger toggle.
5. Partial indexes `exceptions(due_at) WHERE status='open'`, `shortage_flags(due_at) WHERE status='open'`.
6. `actor_write_audit.target_table` CHECK + `'exceptions'`.

Apply wrapper (as `postgres`, port 5432): `BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public; \i migrations/069_exceptions_enforcement.sql; COMMIT;` — ACCESS EXCLUSIVE on `write_tickets` / `actor_write_audit` / the view for milliseconds, bounded by `lock_timeout`.

**Rollback order: 069 before 065 or 061.** The 069 view references the 061 and 065 columns, so
their `DROP COLUMN`s fail while it exists. The down refuses while any ticket is `awaiting_approval`
and (without `SET LOCAL factory_ledger.confirm_exceptions_export='yes'`) while shortage / hold
rows exist; it drops the view CASCADE and recreates it plus its nine 055 dependents from the
2026-10-08 production dump (dependent view OIDs change). The widened audit CHECK and the §4 sweep
are kept (additive / history-only).

## Nightly cron (document only — do NOT create yet)

Same recipe as P1.9's `expire-tickets`: a Railway service from the same repo/branch as FastAPI,
`ENVIRONMENT=production`, `PRODUCTION_DATABASE_HOST=aws-1-us-east-1.pooler.supabase.com`,
`DATABASE_URL=${{FastAPI.DATABASE_URL}}`, dashboard schedule `30 7 * * *` (02:30 ET, after the
plant day closes at 23:59 and before the shift), start command `python scripts/exceptions_sweep.py`.
Expect `Escalated N overdue exception(s) and M shortage flag(s)` in the log; a second run the
same night prints zeros. Staging: same with `ENVIRONMENT=staging` + the staging guard variables.
Anywhere else the script refuses non-local hosts.

## Deviations for the reviewer

1. **Resolve is a direct route, not a ticket.** §7.1 sketches `POST /exceptions/{id}/resolve/prepare`
   → commit with an `XR-` receipt. Michael's A3b brief asks for `resolve (note required; who/when)`,
   so resolve/approve/reject are plain actor-authenticated writes attributed through
   `actor_write_audit` + `resolved_by_actor_id/resolved_at`. A ticketed resolve can be layered on later
   without changing the table.
2. **Shortfall placement**: pinned/override lot → that lot; FIFO → the lot that ran out (last planned
   lot), else the newest lot of the product; **no lot at all → still blocks**. The design says "post
   against the confirmed lot"; the FIFO/newest fallback is the only addition.
3. **Legacy reason codes are accepted on tickets** (translated through the 061 map, translation shown
   in `draft.reason.legacy_code`) — §5.1 says the map is for "any legacy value a client still sends
   during the overlap"; A10 can turn this into a 422.
4. **Held tickets never expire** (brief) rather than "expires_at extended to 7 days" (§7.1).
   `scripts/expire_tickets.py` only touches `prepared`; a re-prepare does not supersede a held ticket.
5. **`adjust_reason` now stores the reason's English label** (`Damage/disposal`) and
   `adjust_reason_es` the Spanish one, with the code in `reason_code`; legacy readers keep seeing
   text, and a future 061-style sweep maps the label back to the code (the "already a code" tier).
6. **Direct legacy routes** keep their behaviour for the master key (G): no new validation; they only
   stamp `reason_code` via the 061 mapping (never NULL while the seed exists).
7. **Highlighted-but-posted corrections open no exception**; the weekly view (A9) reads
   `write_tickets.response->'correction_review'->>'highlighted'` (and the `found` action) — all
   corrections are already there with `reason_code`.
8. **The reference seed is committed to the local test DB** by `tests/conftest.py` (once per session,
   idempotent) because `scripts/dump_prod_schema.sh` refuses data and every ticket correction now
   validates against `correction_reasons`.
9. **(Codex round 1) The `counted` resolution posts a direct adjust transaction**, not an `XR-`/`ADJ-`
   receipt: the counted correction goes through `_adjust_commit_core` inside the resolve transaction
   (reason `physical_count`, `entered_by` = the resolver, notes name the shortage). It is the only path
   that can add stock to a lot with an open shortage, and it is atomic with the close. The
   exception's `detail.resolution` records book-before / counted / delta / transaction_id.
10. **(Codex round 1) The `pre_make_adjust` tag is an `exceptions` row** (`kind='PRE_MAKE_ADJUST'`,
    severity `info`, open until the owner acknowledges) rather than a column on the adjust — the
    ledger is append-only and the weekly view already reads `exceptions`. Window = 30 min of **entry**
    time (`created_at`), same `entered_by_actor_id` (or the same `operator_id` for the shared key),
    positive adjust on any lot of an ingredient the make/pack consumed; `detail.same_lot` says whether
    it was the very lot. Needs migration **070** (kind CHECK + one-tag-per-adjust index).
11. **(Codex round 1) A shared-key preparer never enters a hold**: over 500 lb without a photo the
    master key's commit gets 422 `PHOTO_REQUIRED` (`held: false`) and the ticket stays `prepared`
    until it expires or is committed again with `attachment_ref`. The owner approves people, not keys.

## Coordination with the open PRs

Checked against `origin/feat/fl-assistant` (F1, #90), `origin/feat/order-tickets` (A7, #93) and
`origin/codex/supplier-label-backfill` (#91). None touches the make/pack stock branches,
`validate_lot_deduction`, `choose_inputs`, `_post_prepared_inputs`, the adjust/found INSERTs, the
exception tables or a business-day helper. Expected textual conflicts, all one-liners: the
`import` block and the `ACTOR_WRITE_ALLOWLIST` line and the last lines of `main.py` (F1 adds its
own there), `write_tickets.py` around the commit dispatch and `register_routes` (A7 inserts
there; `execute_commit` was extracted from the block right below A7's insertion point),
`permissions._LABELS` (A7 adds order labels before `backdate_over_14d`; mine sit after `found`),
`tests/schema/schema.sql` pending tail (A7 adds `\ir 067`; F1 `\ir 068`), the isolated-DB fixture
in `tests/test_write_tickets.py` and the actor-scope set in `tests/test_named_actor_writes.py`.
`feat/pin-login` (A11) does not exist on origin yet; the approval route is where its step-up PIN
plugs in (`permissions.require('approve_exception', …)` is the single gate).

## Tests

`tests/test_exceptions_enforcement.py` — reason required / invalid / applies_to / note / sign /
legacy translation; `reason_code` at INSERT for tickets and the four direct-route shapes; the view
columns; highlight matrix (10 % boundary, 500 lb, zero and negative book balance, count-based
product); found 500-lb-only + always weekly; highlight recomputed under lock; photo posts
highlighted; hold → repeat hold → late photo releases once; found over 500 held; owner approval
posts once as the preparer (entered_by, approved_by, replay, reject-after-approve 409, audit row);
reject posts nothing (note required, replay, approve-after-reject 409); stale approval not consumed;
wrong-kind approve / resolve on a hold; **concurrent approvals post once** (dedicated DB);
short make posts + flags + replays with no second flag (+ the DB refuses a duplicate); short pack
against the pinned batch lot; pack from an empty pinned lot; shortfall recomputed under lock
(stock arrived → no flag); no-lot product still blocks; **office 403, floor > 14 d 403,
LOT_NOT_CONFIRMED 422, LOT_MERGED stale — all still block a short make**; never add stock to cover
(adjust +, found, resolve first, then allowed); business-day deadline across weekends / UTC
instant / DST + A5 parity; routes named-actor-only (master, dashboard, bad key, retired actor);
owner-only kinds; list filters / overdue ordering / view / 404 / status validation; sweep
escalates once and the flag follows; sweep script guards; attachment only on corrections;
A2 rows; migration 069 rerun / down (refuses held tickets and unexported rows) / up; the P1.8 sweep.

Results (2026-10-08 ET): fresh local PostgreSQL 17 database built from `tests/schema/schema.sql` (+ pending 069 tail + reference seed): **2,190 Python tests passed, 0 failed** (52 new in `test_exceptions_enforcement.py`); **Node 69/69**. Production untouched; staging written only by the acceptance script below.

**Codex round 1 (2026-10-09)** added 17 tests to the same file, each run against the pre-fix code
first (all 17 failed there — the live approve-vs-photo race hit a real `deadlock detected`) and
after the fixes (all pass): blank/whitespace notes refused on resolve and reject; `counted`
needs `counted_lb`, posts the counted correction atomically (book −6 → counted 2 → +8 adjust,
`physical_count`, entered_by the resolver, shortage flag follows, positive adjust unlocked only
then); `counted` over 500 lb needs a photo; `missing_movement` needs `receipt_number` and the
receipt must put ≥ the short pounds on the SAME lot after the shortage opened (the make's own
receipt, an unrelated receive and a 2 lb receive are all `RECEIPT_NOT_MATCHING` and leave the
cover-up refusal in force; a 10 lb receive onto the lot resolves it); `voided` needs the short
posting voided (`POSTING_STILL_EFFECTIVE` otherwise); `written_off` owner-only and `identified`
needs `lots.identity_status='identified'`; a deactivated / demoted approver is 403 under lock even
though the auth cache still admits the key (approve and reject), nothing posts, hold intact; a
preparer demoted to office → 409 `TICKET_STALE` / `ROLE_NOT_ALLOWED`, nothing posts, approval
works again once the role is restored; a held ticket whose preparer row is gone → 409
`PREPARER_UNKNOWN`, not a 500; the master key's > 500 lb commit → 422 `PHOTO_REQUIRED`, no hold,
ticket stays `prepared`, posts with a photo; **lock order**: a session holding the ticket row can
update the exception row while an approve waits (deterministic, `pg_stat_activity` poll) and a
live approve-vs-photo-commit race returns 200/200, posts once; small `unknown` adjust and found
are highlighted (`REASON_UNKNOWN`); `pre_make_adjust` (31-min-old adjust not tagged, negative not
tagged, other actor not tagged, the +3 lb adjust tagged once across two makes, listed to the owner,
floor cannot acknowledge); migration 070 rerun / down (refuses tagged rows until the export GUC) / up.

Results (2026-10-09 ET): fresh local database `factory_ledger_test_a3b2` from `tests/schema/schema.sql`
(+ pending 069 and 070 tails + reference seed): **2,207 Python tests passed, 0 failed**; **Node 69/69**.

## Staging acceptance

`scripts/check_exceptions_staging.py --apply-migration --output docs/deployments/a3b-staging-receipt.json`
runs the branch's real HTTP routes locally against the guarded staging DB (TestClient without
lifespan: no startup sweeps; URI never printed; temporary floor/owner/office actors deactivated
in `finally`). Evidence: `docs/deployments/a3b-staging-receipt.json` — the three examples Michael
asked for (short make with shortage flag; > 500 lb correction held then approved once; denied
permission attempts). Values are in the PR description.

**Migration 070 on staging (Codex round 1):** `scripts/check_exceptions_staging.py --apply-migration
--migrations-only --output docs/deployments/a3b-staging-070.json` applied `070_pre_make_adjust`
on 2026-10-09 12:53:34Z (069 marker from 01:47:54Z confirmed). No examples were re-run and no
rows were written beyond the CHECK rebuild, the index and the marker. The acceptance script's
shortage resolution now sends `counted_lb: 0` (the counted correction posts atomically), so a future
full run records `shortage_resolution` in the receipt. **Rollback order is now 070 → 069 → 065/061.**

## Codex review round 1 (2026-10-09) — what changed

Verdict was "merge after fixes"; all six items are in, each with regression tests that fail on the
pre-fix code. Every change stays inside `exceptions_enforcement.py` except the two-line shared-key
branch in `write_tickets.execute_commit`, the `reason_code=` argument on both `correction_review`
calls and the `record_pre_make_adjusts` call in `ticket_actions.post`.

1. **Resolution evidence** — `shortage_evidence()`: `counted` (→ `counted_lb`, atomic correction,
   `COUNT_REQUIRED` / `PHOTO_REQUIRED` over 500 lb), `missing_movement` (→ `receipt_number`, same lot,
   ≥ short lb, entered after `opened_at`, still effective; else 409 `RECEIPT_NOT_MATCHING` listing the
   problems), `voided` (short posting `effective_status='voided'`, else 409 `POSTING_STILL_EFFECTIVE`).
   `identified_evidence()` for UNIDENTIFIED_LOT. `written_off` / `waived` require `approve_exception`
   (owner). `ResolveRequest` / `DecisionRequest` strip and refuse blank notes (pydantic validator);
   `ResolveRequest` gains `counted_lb` and `attachment_ref`. `detail.resolution` on the row.
2. **Approval permissions** — `_current_approver()` re-reads the approver `FOR SHARE` (active + role →
   403 `ACTOR_INACTIVE` / `ROLE_NOT_ALLOWED`) in approve and reject; `_current_preparer()` re-reads the
   preparer `FOR SHARE` and checks `permissions.allowed(action, role)` → 409 `TICKET_STALE` with
   `ROLE_NOT_ALLOWED` / `ACTOR_INACTIVE` / `PREPARER_UNKNOWN`, nothing posted, hold kept.
3. **Lock order** — `_held()` now peeks the exception (unlocked; `ticket_id` is immutable), locks the
   **ticket** `FOR UPDATE`, then the exception — the same order as the photo-release commit. Approve
   also replays whenever the ticket is already `committed` (approved or photo-released first).
4. **Shared-key preparers** — `photo_required()`: 422, `held: false`, ticket stays `prepared`.
5. **Design flags** — `correction_review(..., reason_code=)` adds `REASON_UNKNOWN` (§5.1);
   `record_pre_make_adjusts()` + migration 070 (§5 R3); `RESOLUTION_KINDS['PRE_MAKE_ADJUST'] =
   ('acknowledged',)`, owner kind.
6. **FOLLOWUPS P1.13** marked pilot-blocking (A6 must validate `attachment_ref` ownership/existence).

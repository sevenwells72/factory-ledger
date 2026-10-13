# A5 part 2 — supplier-lot correction tickets (P1.15)

Implements the remaining A5 correction flow from FOLLOWUPS P1.15 (formerly the
second P1.11) and design R8, §5/§5.1, §4.3 and §7.1. Built on `f4e51b5`.
The branch is for Claude Code review; do not merge or deploy to production.

## D1 “Resolve → supplier lot” contract

Use the exception's numeric `lot_id`. Authenticate every request with
`X-API-Key: <personal actor key or key-session token>`. Owner, floor and office
are allowed. The shared master/dashboard keys do not get this new correction
route. Caller-supplied actor IDs and supplier IDs are rejected.

1. `POST /lots/{lot_id}/supplier-lot/prepare`:

   ```json
   {
     "supplier_lot_code": "8812-A",
     "reason_code": "data_entry_error",
     "note": "Recovered the supplier packing slip.",
     "client_source": "dashboard"
   }
   ```

   Required: `supplier_lot_code`, `reason_code`. Optional: `note` (2,000 chars),
   `client_source` (default `api`), `occurred_at` or `happened_at`, `backfill`.
   Only the two applicable fixed-list reasons are accepted: `wrong_lot` and
   `data_entry_error`. Other codes, including `unknown` with a note, return
   `422 REASON_CODE_INVALID`; the catalog's applicability is authoritative.

   Codes are trimmed, preserve case, start with an ASCII letter/digit and are
   1–200 characters; remaining characters may be ASCII letters/digits, spaces,
   `. _ / # ( ) + -`. Blank, `N/A`, `UNKNOWN`, other placeholders, punctuation-only
   strings, control characters, non-strings and markup return
   `422 SUPPLIER_LOT_CODE_INVALID`, without issuing a ticket.

   Success uses the unchanged A1 envelope:
   `{ticket, ticket_id, action:"update_supplier_lot", payload_hash, expires_at,
   draft, warnings, blockers, can_commit, actor}`. Display at least
   `draft.product_name`, `lot_code`, `previous_supplier_lot_code`,
   `supplier_lot_code`, `reason`, `note`, and `exception_id` (nullable).
   Enable Record only when `can_commit` is true; show blockers otherwise.
   A missing/merged/inactive lot, inapplicable product or commingled lot gives
   a blocked draft. This simple one-code contract covers purchased-ingredient
   lots with no component rows in `lot_supplier_codes`; it cannot declare a
   multi-component lot fully identified from one label.

2. On explicit Record, `POST /tickets/{ticket}/commit`:

   ```json
   { "payload_hash": "<exact hash from prepare>", "acknowledged_warnings": [] }
   ```

   Keep the ticket and hash for retries. Do not send the code, actor or reason
   again. Use any required warning codes supplied by prepare. The response is:

   ```json
   {
     "success": true,
     "updated": true,
     "lot_id": 123,
     "lot_code": "26-10-13-STGA-001",
     "product_id": 456,
     "product_name": "Example ingredient",
     "previous_supplier_lot_code": "UNKNOWN",
     "supplier_lot_code": "8812-A",
     "identity_status": "identified",
     "identify_by": null,
     "reason_code": "data_entry_error",
     "note": "Recovered the supplier packing slip.",
     "lot_event_id": 789,
     "resolved_exception_ids": [321],
     "entered_by_actor_id": 12,
     "receipt_number": "LOT-261013-001",
     "ticket_id": 654,
     "replayed": false,
     "state_changed": false,
     "entry_timing": { "entered_by": { "id": 12, "name": "Example operator" } }
   }
   ```

   IDs above are illustrative. `entry_timing` also has the standard A1 timing
   fields. `resolved_exception_ids` is empty when no open exception exists.
   No inventory quantity or supplier identity changes; no ledger transaction
   is fabricated. `GET /receipts/{receipt_number}` returns the saved response,
   the lot in `lots`, and `transactions: []`.

3. Show the `LOT-…` receipt and refresh the exceptions list. **No separate
   `/exceptions/{id}/resolve` call is needed** (office is allowed here through
   `update_supplier_lot`, regardless of the generic resolver's floor/owner rule).

Existing A1 behavior: ten-minute expiry (`dashboard` drafts use A1's existing
30-minute expiry); session idle expiry is separate. Another actor gets
`403 TICKET_WRONG_USER`. A bad hash gets `409 TICKET_PAYLOAD_MISMATCH`. Expired
or superseded tickets require a new prepare. Retry after a lost commit response
returns the saved receipt with `replayed:true`. A correction/rename/merge since
prepare returns `409 TICKET_STALE` with `LOT_IDENTITY_CHANGED`/`LOT_MERGED`
inside `detail.blockers`; show the error and prepare a fresh draft.

## Atomicity and attribution

A1 locks the ticket and rechecks the live actor role/active state. Validation
locks product then lot, preserving receive's lock order, and pins the old lot
identity plus the most recent correction event ID. Different correction tickets
serialize on the lot; a stale second draft cannot overwrite the first, even
if the label was changed away and back. Same-ticket races replay one result.

One transaction updates the supplier code, sets `identity_status=identified`,
clears `identify_by`, inserts one append-only `lot_events(event=supplier_lot)`
with old/new identity, fixed reason, event time, authenticated
`entered_by_actor_id` and ticket ID, and resolves the open/escalated
`UNIDENTIFIED_LOT` with `resolution_kind=identified`, `resolved_at`,
`resolved_by_actor_id`, `resolution_note` and `resolution_ticket_id`. A1 saves
the receipt in that transaction. Audit failure rolls everything back.
Later corrections never rewrite a closed exception's resolution evidence.

The existing receive guard remains in force: topping up an unidentified lot
with another unknown code retains the exception and its deadline; supplying a
real code via receive cannot resolve it. A receive prepared before correction
is revalidated and cannot replace the corrected identity. The legacy receive
route also leaves the exception open; direct-route retirement remains A10.

## Migration and validation

`079_supplier_lot_tickets.sql` only widens `write_tickets_action_check` and adds
its marker. Existing actions are preserved. Reuses 064, 065 and 075 columns and
append-only audit protection. Apply before the code, explicitly as app/table
owner with `ON_ERROR_STOP`, `BEGIN`, `SET LOCAL search_path=public`,
`lock_timeout='5s'`, `statement_timeout='60s'`, `COMMIT`. Builder scope is
**staging only**; no startup apply. The local schema fixture contains the same
constraint extension; no production schema read was performed.

The down migration is manual, after reverting the app. It refuses while **any**
supplier-lot ticket exists; retain the additive migration once replay/audit
evidence exists. It never deletes tickets, lot events or exception history.

Run the guarded acceptance with the pinned Python 3.12 environment:

```sh
python scripts/check_supplier_lot_staging.py --apply-migration \
  --output docs/deployments/a5-supplier-lot-staging.json
```

It uses this branch's HTTP routes against the guarded staging database, without
a hosted deployment or startup hooks. Only synthetic product/supplier/actor/lot
rows are used. Credentials remain in memory; the synthetic actor is deactivated
on exit. The output contains receipt IDs and sanitized evidence, never tokens.

## Verified 2026-10-13

Fresh isolated UTF-8 PostgreSQL 17: **2,795 Python passed**, no failures/errors/skips
(68.61 seconds; existing deprecation warnings). **107 Node passed**, no failures
or skips. Focused A5 suite: 50 cases. See [test evidence](a5-supplier-lot-tests.json).

Migration 079 applied only to guarded staging. Local branch HTTP with an office
key session received lot `STG-A5P2-F9454E5513` as UNKNOWN (`RCV-261013-007`), then
posted a top-up (`RCV-261013-008`) that left exception **22** open. Correction
**LOT-261013-001** (ticket **88**) set supplier code
`STG-A5P2-F9454E5513-SUP-8812`, marked the lot identified and resolved exception
**22** as `identified`, with the same actor and ticket. Replay returned the same
receipt; exactly one correction event (**2**) exists, and the correction left
stock at 4 lb. The temporary actor was deactivated. See
[sanitized staging evidence](a5-supplier-lot-staging.json). No hosted deployment
or production access was performed.

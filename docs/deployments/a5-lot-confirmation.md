# A5 — lot confirmation, substitutions, unidentified lots, suppliers

Builder: Codex. Reviewer: Claude Code. Draft; do not merge.
Base: origin/main 1f3f098 (design revision 3.7).

## Claude Code review fixes

Only supplier-sourced ingredient lots receive identity flags, for both receive
and found. Finished/batch/packaging/consumable products, services, internal recipe
outputs and exclusively inventory-excluded ingredients do not open
UNIDENTIFIED_LOT exceptions. A new ingredient needs no prior receipt to qualify.
The shared draft/post policy reads the authoritative product, not a client flag.

Migration **066 is removed from this PR**. Supplier FK columns, the nullable
`suppliers.short_code` column and lot-supplier immutability are schema-only parts
of **064**. A5 works before any label backfill: NULL labels use the resolved
supplier name's existing prefix convention, while identity always uses the ID.
The separate 066 PR must wait for **Michael's supplier cleanup** (Dutch Valley
duplicates / `DUTC Valley` typo). It includes a read-only production label preview;
no production changes are part of this review fix.

Before applying 064, run `scripts/check_unidentified_lots_preapply.py
--database-url-file <protected URI file>`; exit 0 requires no duplicate open or
escalated UNIDENTIFIED_LOT exceptions per lot. The staging apply runner and the
migration itself repeat this check. Investigate duplicates before applying; the
check never deletes or resolves anything.

F1/D2: `last4` is literally the last four characters **including the hyphen**
(e.g. `"-004"`). `pallet` requires the **full lot code echoed as value** and the
latest matching production move within 24 hours. See FOLLOWUPS P1.11:
**A5 part 2 — supplier-lot correction ticket (resolves UNIDENTIFIED_LOT inside FL),
owner Codex, due before Nov 2 pilot.**

Review-fix verification: **1,871 Python tests + 69 JavaScript tests passed**,
zero failures/skips, using a fresh local PostgreSQL 17 database with only
062/063/064 applied (no 066). Production preflight was **read-only** and found
**no duplicate open UNIDENTIFIED_LOT exceptions per lot**. No migration or data
write ran against production; no other worktree was touched.

## Checkpoints

1. Lot confirmations and pallet-move evidence: complete, pushed `7c7857e`; 175 targeted tests passed.
2. Substitutions with reasons: complete, pushed `2c66870`; 122 targeted tests passed.
3. Unidentified lots and seven-business-day deadline: complete, pushed `8189515`; 200 targeted tests passed.
4. Real supplier identity on receipts: complete, pushed `aaaf915`; 335 targeted tests passed. Final follow-up aligns missing/ineligible supplier prepare with HTTP 422.

Commit and push each finished part. Recheck `feat/roles` before shared edits.
A5 owns confirmation validators; A3b owns insufficient-stock behavior.
Direct legacy routes remain unchanged. Migrations are explicit and staging-only.

## Integration baseline

The fetched main implements receive/make/pack/adjust/found tickets only.
Ship and supplier-lot correction tickets are absent. A5 integrates existing
consumption tickets and adds the specified pallet-move ticket; later ticket
adapters can use the same interface-neutral confirmation module.

## Shared hook points

- `ticket_actions.validate`: dispatch `move_lot`; call A5 confirmation validator after stock validation.
- `ticket_actions.post`: dispatch move evidence and save confirmations after core posting.
- `write_tickets`: typed request fields/move route; late-evidence merge before validation;
  recoverable 422 confirmation blockers; non-ledger move result refs/receipt reads.
- `main._make_commit_core`: optional validated `substitutions` argument and named
  formula mapping hook immediately after formula fetch. Legacy callers supply none.
- `ticket_actions.validate/post`: A5 substitution validation, specification and evidence.
- `choose_inputs`, `_post_prepared_inputs` and insufficient-stock branches: unchanged.

The original payload/hash stays immutable. Prepare returns `confirmed:false`
and `LOT_NOT_CONFIRMED` until evidence is supplied. Commit can add only
`lot_confirmations`; it cannot replace lot selections or quantities. Missing or
invalid evidence returns 422 and leaves the ticket prepared for a retry.
`last4` requires exactly four matching characters and a unique suffix among
positive-stock, non-merged lots of that product; `full_code`/`scan` match the full
code. `pallet` uses the full lot code plus the latest move being to production,
within 24 hours. A later storage/staging move invalidates that evidence.

Pallet moves use `/lots/{lot_id}/move/prepare` with `to_location`, `method`
(`last4`, `full_code`, `scan`) and `value`, followed by normal ticket commit.
They create a `LOT` receipt and immutable `lot_moves` row, without ledger stock
movement. Named actors retain attribution; dashboard scope is unchanged.

## Validation and staging evidence

Initial builder complete suite: **1,856 Python tests passed**, zero failures/skips, on a
fresh dedicated local PostgreSQL 17 database (`127.0.0.1:57688/fl_a5_release`).
**69 JavaScript tests passed**, zero failures/skips. All four migrations rerun
idempotently with supplier labels and marker timestamps unchanged. Whitespace
check passes. AST comparison confirms `choose_inputs` and
`_post_prepared_inputs` are identical to origin/main.

Historical builder acceptance: migrations **062, 063, 064 and the original 066 applied to STAGING only** (one transaction,
app/table owner, port 5432, `lock_timeout=5s`, `search_path=public`). The local
branch ran its real HTTP routes against the guarded staging database with
startup migrations/sweeps disabled. The hosted staging service, production,
other worktrees, roles branch and direct endpoints were not changed.

Final reference: `STG-A5-58F9C7C8DB95`; temporary actor `1000000008`, deactivated
in `finally`. Fixtures and receipts are synthetic acceptance stock, retained.
The supplier is the existing real staging catalog vendor **Dutch Gold Honey,
ID 13**, explicitly selected from `/resolve kind=supplier` candidates.

| Check | Final receipt/evidence |
|---|---|
| Receive | **RCV-261008-004**, supplier 13 on transaction and new lot at INSERT |
| Confirmed make | **MK-261008-005**, lot **26-10-08-DUTB-004**, lot ID 1000000018, typed last four **-004** |
| Pallet move | LOT-261008-002, production move with full lot code |
| Recorded substitution | MK-261008-006, `acceptance_trial` reason |
| Unidentified receive | RCV-261008-006, exception due **2026-10-19 23:59 America/New_York** |
| Refusals | Unconfirmed make 422; substitution without reason 422; missing real supplier 422 |
| Idempotency | Identical make replay returned the same saved receipt |

Full receipt JSON: [final evidence](a5-staging-receipt-final.json).
Earlier acceptance evidence: [initial evidence](a5-staging-receipt.json).
Reproducible runner: `scripts/check_lot_confirmation_staging.py` (URI never
printed; it reads only the protected staging URI file). No hosted deployment,
production query/migration, PR merge, or user notification was performed.

Deployment prerequisite: apply 062/063/064 explicitly before this application
code, after the duplicate-exception preflight. 066 is NOT an A5 prerequisite.
The original staging-only 066 application above remains historical evidence; this
review fix does not undo staging labels or apply anything to production. They are additive; no historical transaction/lot supplier inference or
ledger rewrite occurs. Roll back application code first and retain additive
schema/evidence; do not drop recorded confirmations, moves or substitutions.
A2 migration 065 is independent. Claude Code review is still required.

## Substitution contract

Make accepts `substitutions: [{ingredient_product_id, substitute_product_id,
lot_id, reason_code, note}]`. The substitute takes the original pounds; the
master formula is unchanged. It must be an active stock ingredient/batch, with
a matching lot, and needs its own confirmation. Excluded/duplicate/self/output
substitutions are blocked. Multiple requirements for one replacement aggregate
and use one selected lot. Manual `excluded_ingredients` require a top-level
`reason_code` (optional `note`); automatic recipe exclusions retain their policy.
Reasons here are operator-supplied nonblank codes; the §5.1 fixed correction
catalog explicitly applies to corrections, not make substitutions.
`transaction_substitutions` stores original/replacement/lot/reason/note/actor
atomically, including null replacement for a manual exclusion. Receipt JSON
exposes the evidence; ledger and ingredient trace consume the actual substitute.

## Unidentified lots

For supplier-sourced ingredients only, blank, `N/A`, `NA`, and `UNKNOWN`
supplier lot codes flag a receive as unidentified. Internal lot labels never stand in for supplier lot evidence.
Commingled receipts need a real code on every entry. Found tickets use the same
ingredient-only flagging (optional `supplier_lot_code`). Non-supplier-sourced
products remain NULL/unassessed and never open UNIDENTIFIED_LOT exceptions.
History remains NULL/unassessed.
The draft shows `UNIDENTIFIED — resolve by <date>` without blocking posting.
The commit creates one open 061 `UNIDENTIFIED_LOT` exception per lot, owned by
active floor actor Arturo when present, otherwise the first active floor actor
(or unassigned if no active floor actor exists). It links ticket/receipt/transaction.

The clock starts at `transactions.created_at` (entry, never backdated happened
or lot received time). Count seven Mon–Fri days after the local entry date,
skipping weekends, not holidays; deadline is 23:59 America/New_York. Friday
Oct 9 => Tuesday Oct 20, 2026. DST uses the deadline date's local offset.
Top-ups preserve the original exception/deadline. Identifying an already flagged
lot requires the future supplier-lot correction ticket; another receive cannot
silently clear its exception. The correction ticket is absent from this baseline.
Shared hooks: `write_tickets.validate_receive` identity draft, receive commit
identity evidence, and equivalent found hooks in `ticket_actions`.

## Supplier identity

Receive requires an active real `supplier_id` eligible in A4 `/resolve`.
Missing, inactive and pseudo-supplier IDs return HTTP 422
`SUPPLIER_REQUIRED` before a ticket is issued. Commit rechecks eligibility. Names and lot prefixes never select the supplier.
`suppliers.short_code` is a nullable unique four-letter display label. Its
backfill and future-insert allocator belong to the deferred 066 PR, after
Michael's cleanup. Before 066, a NULL short code uses the resolved supplier
name's existing display prefix; duplicate prefixes are allowed because they
are never identities. An assigned short code takes precedence. A client prefix
override cannot choose a different supplier or override its display policy.
Explicit physical lot labels remain allowed and carry no supplier identity.

New transactions, lots and commingled supplier-code entries store supplier FKs
at INSERT. Lot supplier IDs are immutable; historical NULL lots stay unassessed
when topped up, while the new receipt transaction records the chosen supplier.
Known lots cannot be topped up from another supplier. Expected-receipt matching
is pinned by the same supplier ID. Receipt reads expose supplier FKs/names,
confirmations and substitutions. Legacy direct endpoints keep their behavior.

Shared supplier hooks: `write_tickets.supplier`, `validate_receive`, commit
and `receipt_detail`; `main.find_or_create_lot` optional supplier INSERT/check;
`main._receive_commit_core` optional supplier argument and receive/commingled
INSERT columns. No stock-check, `choose_inputs`, or `_post_prepared_inputs` edit.

## A2 coordination

Before shared edits, fetched and inspected local `feat/roles`. Initially at
1f3f098; latest inspected commit cbe9668 adds A2 permissions and actor IDs.
A5 supplier schema is in **064**; **066** is deferred to a separate post-cleanup
PR, leaving A2's **065_entered_by** untouched.
The receive transaction INSERT is a known merge point: preserve BOTH A2
`entered_by_actor_id` and A5 `supplier_id`, with matching VALUES/parameters.
**feat/roles merge trap (Claude Code review): after resolving the receive INSERT
column list (14 columns), the VALUES line needs a 14th `%s`.** Check the final
column/value/parameter counts together; resolving only the columns will fail
at runtime. Preserve both attribution and supplier provenance.
A2 permission/backdating hooks must remain before A5 validation/posting.
A2's cbe9668 role map already includes `move_lot` for named roles/master; preserve it when integrated. No A2 checkout was edited and no A2 code was merged into this PR.

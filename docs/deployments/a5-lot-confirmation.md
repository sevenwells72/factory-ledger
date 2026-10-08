# A5 — lot confirmation, substitutions, unidentified lots, suppliers

Builder: Codex. Reviewer: Claude Code. Draft; do not merge.
Base: origin/main 1f3f098 (design revision 3.7).

## Checkpoints

1. Lot confirmations and pallet-move evidence: complete; 175 targeted tests passed.
2. Substitutions with reasons: complete; 122 confirmation/substitution/A1 tests passed.
3. Unidentified lots and seven-business-day deadline: complete; 200 targeted tests passed.
4. Real supplier identity on receipts: complete; 335 A5/ticket/resolver tests passed.

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

Pending. No production access, deployment, or merge authorized.

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

Blank, `N/A`, `NA`, and `UNKNOWN` supplier lot codes flag a receive as
unidentified. Internal lot labels never stand in for supplier lot evidence.
Commingled receipts need a real code on every entry. Found tickets use the same
flagging (optional `supplier_lot_code`). History remains NULL/unassessed.
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
Missing, inactive and pseudo-supplier IDs issue a blocked draft with
`SUPPLIER_REQUIRED`. Names and lot prefixes never select the supplier.
`suppliers.short_code` is a unique four-letter display label, seeded from the
name's token when available and assigned collision-safe alternatives otherwise.
New real suppliers receive a label at INSERT. A ticket-generated label uses the
resolved supplier's short code; a client prefix override does not override it.
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
A5 supplier migration is **066**, leaving A2's **065_entered_by** untouched.
The receive transaction INSERT is a known merge point: preserve BOTH A2
`entered_by_actor_id` and A5 `supplier_id`, with matching VALUES/parameters.
A2 permission/backdating hooks must remain before A5 validation/posting.
A2's role map must include `move_lot` for the intended inventory roles when
integrated. No A2 checkout was edited and no A2 code was merged into this PR.
